"""
RF-DETR -> CoreML conversion without YAML config.

This script is designed for checkpoints like:
  weights/rf-detr-xxlarge.pth

The checkpoint is expected to contain a state dict under "model" (or other common keys).
Model architecture is instantiated from the rfdetr Python package, then weights are loaded,
wrapped for inference output compatibility, and exported to CoreML.

Example:
  python export_rf_detr_coreml.py \
    --checkpoint weights/rf-detr-xxlarge.pth \
    --model_class RFDETR2XLarge \
    --input_size 640 \
    --output weights/rf-detr-xxlarge.mlpackage
"""

import argparse
import inspect
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


COMPUTE_UNIT_MAP = {
    "all": ct.ComputeUnit.ALL,
    "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
    "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
    "cpu_only": ct.ComputeUnit.CPU_ONLY,
}


class CoreMLFriendlyMultiheadAttention(nn.Module):
    """
    Export-friendly replacement for nn.MultiheadAttention(batch_first=True).
    """

    def __init__(self, mha: nn.MultiheadAttention):
        super().__init__()
        if not mha.batch_first:
            raise ValueError("Only batch_first=True MultiheadAttention is supported")

        self.embed_dim = mha.embed_dim
        self.num_heads = mha.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")

        has_bias = mha.in_proj_bias is not None
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=has_bias)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=has_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=has_bias)

        w_q, w_k, w_v = mha.in_proj_weight.detach().chunk(3, dim=0)
        self.q_proj.weight.data.copy_(w_q)
        self.k_proj.weight.data.copy_(w_k)
        self.v_proj.weight.data.copy_(w_v)

        if has_bias:
            b_q, b_k, b_v = mha.in_proj_bias.detach().chunk(3, dim=0)
            self.q_proj.bias.data.copy_(b_q)
            self.k_proj.bias.data.copy_(b_k)
            self.v_proj.bias.data.copy_(b_v)

        self.out_proj = nn.Linear(
            self.embed_dim,
            self.embed_dim,
            bias=mha.out_proj.bias is not None,
        )
        self.out_proj.weight.data.copy_(mha.out_proj.weight.detach())
        if mha.out_proj.bias is not None:
            self.out_proj.bias.data.copy_(mha.out_proj.bias.detach())

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        return x.reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask=None,
        key_padding_mask=None,
        need_weights=True,
        average_attn_weights=True,
        is_causal=False,
    ):
        if key_padding_mask is not None:
            raise ValueError("key_padding_mask is not supported in export-friendly attention")

        q = self._reshape_heads(self.q_proj(query))
        k = self._reshape_heads(self.k_proj(key))
        v = self._reshape_heads(self.v_proj(value))

        attn_mask_t = None
        if attn_mask is not None:
            if attn_mask.ndim == 2:
                attn_mask_t = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.ndim == 3:
                attn_mask_t = attn_mask.unsqueeze(1)
            else:
                attn_mask_t = attn_mask
            attn_mask_t = attn_mask_t.to(device=q.device)

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask_t,
            dropout_p=0.0,
            is_causal=is_causal,
        )
        attn_output = attn_output.transpose(1, 2).reshape(
            query.shape[0],
            query.shape[1],
            self.embed_dim,
        )
        attn_output = self.out_proj(attn_output)

        if not need_weights:
            return attn_output, None

        bsz = query.shape[0]
        tgt_len = query.shape[1]
        src_len = key.shape[1]
        if average_attn_weights:
            attn_weights = torch.zeros(bsz, tgt_len, src_len, device=query.device, dtype=query.dtype)
        else:
            attn_weights = torch.zeros(
                bsz,
                self.num_heads,
                tgt_len,
                src_len,
                device=query.device,
                dtype=query.dtype,
            )
        return attn_output, attn_weights


class CoreMLFriendlyMSDeformAttn(nn.Module):
    """
    Export-friendly replacement for RF-DETR MSDeformAttn that avoids rank-6 tensors.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.d_model = int(module.d_model) # type: ignore
        self.n_levels = int(module.n_levels) # type: ignore
        self.n_heads = int(module.n_heads) # type: ignore
        self.n_points = int(module.n_points) # type: ignore
        self.im2col_step = int(getattr(module, "im2col_step", 64))

        self.sampling_offsets = nn.Linear(
            self.d_model,
            self.n_heads * self.n_levels * self.n_points * 2,
            bias=True,
        )
        self.attention_weights = nn.Linear(
            self.d_model,
            self.n_heads * self.n_levels * self.n_points,
            bias=True,
        )
        self.value_proj = nn.Linear(self.d_model, self.d_model, bias=True)
        self.output_proj = nn.Linear(self.d_model, self.d_model, bias=True)

        self.sampling_offsets.weight.data.copy_(module.sampling_offsets.weight.detach()) # type: ignore
        self.sampling_offsets.bias.data.copy_(module.sampling_offsets.bias.detach()) # type: ignore
        self.attention_weights.weight.data.copy_(module.attention_weights.weight.detach()) # type: ignore
        self.attention_weights.bias.data.copy_(module.attention_weights.bias.detach()) # type: ignore
        self.value_proj.weight.data.copy_(module.value_proj.weight.detach()) # type: ignore
        self.value_proj.bias.data.copy_(module.value_proj.bias.detach()) # type: ignore
        self.output_proj.weight.data.copy_(module.output_proj.weight.detach()) # type: ignore
        self.output_proj.bias.data.copy_(module.output_proj.bias.detach()) # type: ignore

    @property
    def _d_per_head(self) -> int:
        return self.d_model // self.n_heads

    def _level_hw(self, input_spatial_shapes: torch.Tensor, level: int) -> Tuple[int, int]:
        # Export runs at fixed input size, so these shape values are constants.
        h = int(input_spatial_shapes[level, 0].item())
        w = int(input_spatial_shapes[level, 1].item())
        return h, w

    def _level_slice(
        self,
        input_level_start_index: torch.Tensor,
        level: int,
        total_len: int,
    ) -> Tuple[int, int]:
        start = int(input_level_start_index[level].item())
        if level + 1 < self.n_levels:
            end = int(input_level_start_index[level + 1].item())
        else:
            end = total_len
        return start, end

    def forward(
        self,
        query,
        reference_points,
        input_flatten,
        input_spatial_shapes,
        input_level_start_index,
        input_padding_mask=None,
    ):
        from rfdetr.utilities.tensors import _bilinear_grid_sample

        n_batch, len_q, _ = query.shape
        _, len_in, _ = input_flatten.shape

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))

        sampling_offsets = self.sampling_offsets(query).view(
            n_batch,
            len_q,
            self.n_heads,
            self.n_levels * self.n_points,
            2,
        )
        attention_weights = self.attention_weights(query).view(
            n_batch,
            len_q,
            self.n_heads,
            self.n_levels * self.n_points,
        )
        attention_weights = F.softmax(attention_weights, dim=-1)

        value = value.transpose(1, 2).contiguous().view(
            n_batch,
            self.n_heads,
            self._d_per_head,
            len_in,
        )

        output_acc = None
        for lid in range(self.n_levels):
            h_l, w_l = self._level_hw(input_spatial_shapes, lid)
            start, end = self._level_slice(input_level_start_index, lid, len_in)

            value_l = value[..., start:end].contiguous().view(
                n_batch * self.n_heads,
                self._d_per_head,
                h_l,
                w_l,
            )

            offset_l = sampling_offsets[
                ..., lid * self.n_points : (lid + 1) * self.n_points, :
            ]

            if reference_points.shape[-1] == 2:
                ref_xy = reference_points[:, :, lid, :].unsqueeze(2).unsqueeze(3)
                normalizer = torch.stack(
                    [
                        input_spatial_shapes[lid, 1],
                        input_spatial_shapes[lid, 0],
                    ]
                ).to(dtype=offset_l.dtype, device=offset_l.device)
                sampling_locations_l = ref_xy + offset_l / normalizer.view(1, 1, 1, 1, 2)
            elif reference_points.shape[-1] == 4:
                ref_l = reference_points[:, :, lid, :]
                ref_xy = ref_l[..., :2].unsqueeze(2).unsqueeze(3)
                ref_wh = ref_l[..., 2:].unsqueeze(2).unsqueeze(3)
                sampling_locations_l = ref_xy + offset_l / float(self.n_points) * ref_wh * 0.5
            else:
                raise ValueError(
                    "Last dim of reference_points must be 2 or 4, "
                    f"but got {reference_points.shape[-1]}"
                )

            sampling_grid_l = (2.0 * sampling_locations_l - 1.0).transpose(1, 2).flatten(0, 1)

            sampled_l = _bilinear_grid_sample(
                value_l,
                sampling_grid_l,
                padding_mode="zeros",
                align_corners=False,
            )

            attn_l = attention_weights[
                ..., lid * self.n_points : (lid + 1) * self.n_points
            ]
            attn_l = attn_l.transpose(1, 2).reshape(
                n_batch * self.n_heads,
                1,
                len_q,
                self.n_points,
            )

            weighted_l = (sampled_l * attn_l).sum(-1)
            output_acc = weighted_l if output_acc is None else output_acc + weighted_l

        output = output_acc.view(n_batch, self.n_heads * self._d_per_head, len_q) # type: ignore
        output = output.transpose(1, 2).contiguous()
        output = self.output_proj(output)
        return output


def replace_multihead_attention_for_export(module: nn.Module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.MultiheadAttention):
            setattr(module, name, CoreMLFriendlyMultiheadAttention(child))
        else:
            replace_multihead_attention_for_export(child)


def _looks_like_ms_deform_attn(module: nn.Module) -> bool:
    return (
        module.__class__.__name__ == "MSDeformAttn"
        and hasattr(module, "sampling_offsets")
        and hasattr(module, "attention_weights")
        and hasattr(module, "value_proj")
        and hasattr(module, "output_proj")
        and hasattr(module, "n_levels")
        and hasattr(module, "n_heads")
        and hasattr(module, "n_points")
        and hasattr(module, "d_model")
    )


def replace_ms_deform_attn_for_export(module: nn.Module):
    for name, child in list(module.named_children()):
        if _looks_like_ms_deform_attn(child):
            setattr(module, name, CoreMLFriendlyMSDeformAttn(child))
        else:
            replace_ms_deform_attn_for_export(child)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert RF-DETR checkpoint to CoreML")
    parser.add_argument(
        "--checkpoint",
        default="weights/rf-detr-xxlarge.pth",
        help="Path to RF-DETR .pth checkpoint",
    )
    parser.add_argument(
        "--model_class",
        default="RFDETR2XLarge",
        help="Class name from rfdetr package (e.g., RFDETR2XLarge)",
    )
    parser.add_argument(
        "--constructor_kwargs",
        default="{}",
        help="JSON kwargs passed to model class constructor",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=-1,
        help="Number of classes. Use -1 to infer from checkpoint",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=640,
        help="Square input size for export",
    )
    parser.add_argument(
        "--backbone_stride",
        type=int,
        default=40,
        help="Backbone stride divisibility requirement for input size.",
    )
    parser.add_argument(
        "--no_auto_adjust_input_size",
        action="store_true",
        help="Fail instead of auto-adjusting input size to stride requirements.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output .mlpackage path (default: checkpoint stem + .mlpackage)",
    )
    parser.add_argument(
        "--compute_precision",
        choices=["float16", "float32"],
        default="float16",
        help="CoreML compute precision",
    )
    parser.add_argument(
        "--compute_units",
        choices=["all", "cpu_and_gpu", "cpu_and_ne", "cpu_only"],
        default="cpu_and_gpu",
        help="CoreML runtime compute units",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "mps"],
        default="cpu",
        help="Device used for PyTorch export stage",
    )
    parser.add_argument(
        "--strict_load",
        action="store_true",
        help="Use strict checkpoint loading",
    )
    parser.add_argument(
        "--no_mha_patch",
        action="store_true",
        help="Disable nn.MultiheadAttention replacement",
    )
    parser.add_argument(
        "--no_msda_patch",
        action="store_true",
        help="Disable MSDeformAttn replacement",
    )
    parser.add_argument(
        "--no_bicubic_patch",
        action="store_true",
        help="Disable export-time patch that replaces bicubic resize with bilinear.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Load exported CoreML model after conversion",
    )
    parser.add_argument(
        "--export_backend",
        choices=["auto", "torchscript", "torch_export"],
        default="torchscript",
        help=(
            "Backend for PyTorch graph export. 'torchscript' is often more "
            "compatible with coremltools for complex models."
        ),
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run latency benchmark after conversion",
    )
    parser.add_argument(
        "--benchmark_runs",
        type=int,
        default=50,
        help="Number of timed runs for each benchmark backend",
    )
    parser.add_argument(
        "--benchmark_compute_units",
        default="ALL,CPU_AND_GPU,CPU_AND_NE,CPU_ONLY",
        help=(
            "Comma-separated runtime compute units to benchmark against the saved model. "
            "Order matters. Example: CPU_AND_GPU,CPU_ONLY"
        ),
    )
    return parser.parse_args()


def _is_state_dict(value: Any) -> bool:
    return isinstance(value, dict) and bool(value) and all(torch.is_tensor(v) for v in value.values())


def _normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    prefixes = ["module.", "model."]
    out = dict(state_dict)
    for prefix in prefixes:
        keys = list(out.keys())
        if not keys:
            break
        prefixed = sum(1 for k in keys if k.startswith(prefix))
        if prefixed > len(keys) * 0.5:
            out = {k[len(prefix) :]: v for k, v in out.items()}
    return out


def checkpoint_to_state_dict(checkpoint_path: Path) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    metadata: Dict[str, Any] = {}

    if _is_state_dict(checkpoint):
        state_dict = checkpoint
    elif isinstance(checkpoint, dict):
        metadata["checkpoint_keys"] = list(checkpoint.keys())

        candidates = []
        if "model" in checkpoint:
            candidates.append(checkpoint["model"])
        if "state_dict" in checkpoint:
            candidates.append(checkpoint["state_dict"])
        if "ema" in checkpoint and isinstance(checkpoint["ema"], dict):
            ema = checkpoint["ema"]
            if "module" in ema:
                candidates.append(ema["module"])

        state_dict = None
        for candidate in candidates:
            if _is_state_dict(candidate):
                state_dict = candidate
                break

        if state_dict is None:
            raise ValueError(
                "Could not find a valid state_dict in checkpoint. "
                "Expected top-level state_dict or keys like 'model'/'state_dict'."
            )
    else:
        raise ValueError(f"Unsupported checkpoint type: {type(checkpoint)}")

    state_dict = _normalize_state_dict_keys(state_dict)
    return state_dict, metadata


def infer_num_classes_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Optional[int]:
    candidate_keys = [
        "class_embed.weight",
        "transformer.enc_out_class_embed.0.weight",
        "cls_embed.weight",
        "head.class_embed.weight",
    ]
    for key in candidate_keys:
        tensor = state_dict.get(key)
        if tensor is not None and tensor.ndim >= 2:
            raw_classes = int(tensor.shape[0])
            if key in {"class_embed.weight", "transformer.enc_out_class_embed.0.weight"} and raw_classes > 1:
                # DETR-style heads typically include one extra "no object" class.
                return raw_classes - 1
            return raw_classes

    for key, tensor in state_dict.items():
        key_lower = key.lower()
        if "class_embed" in key_lower and tensor.ndim >= 2:
            raw_classes = int(tensor.shape[0])
            return raw_classes - 1 if raw_classes > 1 else raw_classes

    return None


def _import_rfdetr_class(model_class: str):
    try:
        import rfdetr
    except ImportError as exc:
        raise ImportError(
            "Could not import 'rfdetr'. Install it in your active environment first, "
            "for example: pip install rfdetr"
        ) from exc

    if not hasattr(rfdetr, model_class):
        available = sorted(name for name in dir(rfdetr) if "RFDETR" in name)
        raise AttributeError(
            f"Class '{model_class}' not found in rfdetr. Available RFDETR classes: {available}"
        )

    return getattr(rfdetr, model_class)


def _build_constructor_attempts(
    cls,
    num_classes: Optional[int],
    constructor_kwargs: Dict[str, Any],
) -> Iterable[Dict[str, Any]]:
    attempts = []
    if constructor_kwargs:
        attempts.append(dict(constructor_kwargs))

    attempts.append({})

    sig = inspect.signature(cls.__init__)
    heuristic: Dict[str, Any] = {}
    for name, param in sig.parameters.items():
        if name == "self":
            continue

        lname = name.lower()
        if num_classes is not None and lname in {"num_classes", "n_classes", "classes"}:
            heuristic[name] = int(num_classes)
        elif lname in {"pretrained", "use_pretrained", "load_pretrained", "download_pretrained"}:
            heuristic[name] = False
        elif "pretrain" in lname and "weight" in lname:
            heuristic[name] = None
        elif "checkpoint" in lname and (param.default is inspect._empty or param.default is None):
            heuristic[name] = None
        elif lname == "device":
            heuristic[name] = "cpu"

    attempts.append(heuristic)

    if num_classes is not None:
        required_fill = dict(heuristic)
        for name, param in sig.parameters.items():
            if name == "self" or param.default is not inspect._empty:
                continue
            lname = name.lower()
            if lname in {"num_classes", "n_classes", "classes"}:
                required_fill[name] = int(num_classes)
        attempts.append(required_fill)

    seen = set()
    for kwargs in attempts:
        key = tuple(sorted(kwargs.items()))
        if key in seen:
            continue
        seen.add(key)
        yield kwargs


def instantiate_model_object(
    model_class: str,
    num_classes: Optional[int],
    constructor_kwargs: Dict[str, Any],
):
    cls = _import_rfdetr_class(model_class)

    errors = []
    for kwargs in _build_constructor_attempts(cls, num_classes, constructor_kwargs):
        try:
            instance = cls(**kwargs)
            return instance, kwargs
        except Exception as exc:
            errors.append((kwargs, repr(exc)))

    error_lines = [f"kwargs={kwargs} -> {err}" for kwargs, err in errors]
    raise RuntimeError(
        "Failed to instantiate RF-DETR model class. Tried:\n"
        + "\n".join(error_lines)
    )


def extract_torch_module(model_obj: Any) -> nn.Module:
    if isinstance(model_obj, nn.Module):
        return model_obj

    def _enqueue_child(queue, value):
        if value is None:
            return
        queue.append(value)

    queue = [model_obj]
    visited = set()

    preferred_attrs = (
        "model",
        "module",
        "net",
        "detector",
        "inference_model",
        "wrapped_model",
    )
    no_arg_methods = (
        "get_torch_model",
        "as_torch_module",
        "unwrap_model",
        "unwrap",
    )

    while queue:
        current = queue.pop(0)
        current_id = id(current)
        if current_id in visited:
            continue
        visited.add(current_id)

        if isinstance(current, nn.Module):
            return current

        for method_name in no_arg_methods:
            method = getattr(current, method_name, None)
            if callable(method):
                try:
                    _enqueue_child(queue, method())
                except TypeError:
                    # Method may require args; skip silently.
                    pass
                except Exception:
                    pass

        for attr in preferred_attrs:
            try:
                _enqueue_child(queue, getattr(current, attr, None))
            except Exception:
                pass

        raw_dict = getattr(current, "__dict__", None)
        if isinstance(raw_dict, dict):
            for value in raw_dict.values():
                _enqueue_child(queue, value)

    raise TypeError(
        "Could not find torch.nn.Module inside object graph of type "
        f"{type(model_obj)}"
    )


def load_weights_flexible(model: nn.Module, state_dict: Dict[str, torch.Tensor], strict: bool = False):
    if strict:
        load_msg = model.load_state_dict(state_dict, strict=True)
        return load_msg, []

    model_state = model.state_dict()
    filtered_state_dict: Dict[str, torch.Tensor] = {}
    dropped_mismatch = []

    for key, value in state_dict.items():
        if key not in model_state:
            continue
        if model_state[key].shape != value.shape:
            dropped_mismatch.append((key, tuple(value.shape), tuple(model_state[key].shape)))
            continue
        filtered_state_dict[key] = value

    load_msg = model.load_state_dict(filtered_state_dict, strict=False)
    return load_msg, dropped_mismatch


def _resolve_output_tensors(output: Any) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    if isinstance(output, dict):
        if "pred_boxes" in output and "pred_logits" in output:
            return output["pred_boxes"], output["pred_logits"], True
        if "boxes" in output and "scores" in output:
            return output["boxes"], output["scores"], False

        for value in output.values():
            try:
                return _resolve_output_tensors(value)
            except Exception:
                continue

    if isinstance(output, (tuple, list)):
        if len(output) >= 2 and torch.is_tensor(output[0]) and torch.is_tensor(output[1]):
            second = output[1]
            min_val = float(second.detach().min().cpu())
            max_val = float(second.detach().max().cpu())
            is_logits = not (0.0 <= min_val and max_val <= 1.0)
            return output[0], output[1], is_logits

        for value in output:
            try:
                return _resolve_output_tensors(value)
            except Exception:
                continue

    raise RuntimeError(
        "Could not resolve output tensors. Expected keys like pred_boxes/pred_logits or boxes/scores."
    )


def _run_with_mode(model: nn.Module, images: torch.Tensor, mode: str):
    if mode == "tensor":
        return model(images)
    if mode == "list_chw":
        return model([images[0]])
    if mode == "dict_images":
        return model({"images": images})
    if mode == "dict_image":
        return model({"image": images})
    raise ValueError(f"Unsupported call mode: {mode}")


def detect_forward_mode_and_output(model: nn.Module, input_size: int, device: str):
    dummy = torch.zeros(1, 3, input_size, input_size, device=device)
    modes = ["tensor", "list_chw", "dict_images", "dict_image"]
    errors = []

    with torch.no_grad():
        for mode in modes:
            try:
                out = _run_with_mode(model, dummy, mode)
                boxes, values, values_are_logits = _resolve_output_tensors(out)
                if boxes.ndim == 2:
                    boxes = boxes.unsqueeze(0)
                if values.ndim == 2:
                    values = values.unsqueeze(0)
                return mode, values_are_logits, tuple(boxes.shape), tuple(values.shape)
            except Exception as exc:
                errors.append((mode, repr(exc)))

    joined = "\n".join(f"  {mode}: {err}" for mode, err in errors)
    raise RuntimeError(f"Could not detect model forward mode/output format.\n{joined}")


def resolve_effective_input_size(
    requested_size: int,
    backbone_stride: int,
    no_auto_adjust_input_size: bool,
) -> int:
    if requested_size <= 0:
        raise ValueError("input_size must be > 0")
    if backbone_stride <= 0:
        raise ValueError("backbone_stride must be > 0")

    if requested_size % backbone_stride == 0:
        return requested_size

    adjusted = ((requested_size + backbone_stride - 1) // backbone_stride) * backbone_stride
    if no_auto_adjust_input_size:
        raise ValueError(
            "Input size must be divisible by backbone stride. "
            f"Got input_size={requested_size}, backbone_stride={backbone_stride}. "
            f"Try input_size={adjusted} or pass --no_auto_adjust_input_size off."
        )

    print(
        "Adjusted input size to satisfy backbone stride requirement: "
        f"{requested_size} -> {adjusted} (stride={backbone_stride})"
    )
    return adjusted


def patch_interpolate_for_export(disabled: bool):
    if disabled:
        return lambda: None

    original_interpolate = F.interpolate

    def patched_interpolate(
        input,
        size=None,
        scale_factor=None,
        mode="nearest",
        align_corners=None,
        recompute_scale_factor=None,
        antialias=False,
    ):
        patched_mode = mode
        if mode == "bicubic":
            patched_mode = "bilinear"

        return original_interpolate(
            input,
            size=size,
            scale_factor=scale_factor,
            mode=patched_mode,
            align_corners=align_corners,
            recompute_scale_factor=recompute_scale_factor,
            antialias=False,
        )

    F.interpolate = patched_interpolate

    def restore():
        F.interpolate = original_interpolate

    return restore


def patch_coremltools_tensor_inplace_copy():
    """
    Work around coremltools torch_tensor_assign scalar-shape mismatch.

    Some models emit 1-element updates with shape (1,) for slice assignment,
    while torch_tensor_assign expects scalar shape [] for squeeze-style assign.
    """

    import coremltools.converters.mil.frontend.torch.ops as ct_ops
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.frontend.torch.torch_op_registry import _TORCH_OPS_REGISTRY

    original_impl = ct_ops._internal_op_tensor_inplace_copy
    original_registry = _TORCH_OPS_REGISTRY.get_func("_internal_op_tensor_inplace_copy")

    def patched_internal_op_tensor_inplace_copy(context, node):
        data = context[node.inputs[0]]
        updates = context[node.inputs[1]]

        updates_shape = getattr(updates, "shape", None)
        if updates_shape is not None and len(updates_shape) == 1 and updates_shape[0] == 1:
            updates = mb.squeeze(x=updates, axes=[0]) # type: ignore

        begin, end, stride, begin_mask, end_mask, squeeze_mask = ct_ops._get_slice_params(
            context,
            data,
            node.inputs[2:],
        )

        data, updates = ct_ops.promote_input_dtypes([data, updates])
        updated_x = ct_ops._translate_torch_tensor_assign(
            x=data,
            updates=updates,
            begin=begin,
            end=end,
            stride=stride,
            begin_mask=begin_mask,
            end_mask=end_mask,
            squeeze_mask=squeeze_mask,
            name=node.name,
        )
        context.add(updated_x)

    ct_ops._internal_op_tensor_inplace_copy = patched_internal_op_tensor_inplace_copy
    _TORCH_OPS_REGISTRY.set_func_by_name(
        patched_internal_op_tensor_inplace_copy,
        "_internal_op_tensor_inplace_copy",
    )

    def restore():
        ct_ops._internal_op_tensor_inplace_copy = original_impl
        if original_registry is not None:
            _TORCH_OPS_REGISTRY.set_func_by_name(
                original_registry,
                "_internal_op_tensor_inplace_copy",
            )

    return restore


def patch_coremltools_meshgrid():
    """
    Work around coremltools meshgrid converter rejecting non-1D tensor inputs.

    Some traced graphs produce meshgrid inputs with extra singleton dimensions;
    flattening them to 1D preserves semantics for coordinate generation.
    """

    import coremltools.converters.mil.frontend.torch.ops as ct_ops
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.frontend.torch.torch_op_registry import _TORCH_OPS_REGISTRY

    original_meshgrid = ct_ops.meshgrid
    original_meshgrid_reg = _TORCH_OPS_REGISTRY.get_func("meshgrid")
    original_meshgrid_indexing_reg = _TORCH_OPS_REGISTRY.get_func("meshgrid.indexing")

    def patched_meshgrid(context, node):
        inputs = ct_ops._get_inputs(context, node, expected=[1, 2])
        nargs = len(inputs)

        tensor_inputs = inputs[0]
        indexing = inputs[1].val if nargs > 1 else "ij"
        indexing = ct_ops._get_kwinputs(context, node, "indexing", default=[indexing])[0] # type: ignore

        if not isinstance(tensor_inputs, (list, tuple)) or len(tensor_inputs) < 2:
            raise ValueError("Requires >= 2 tensor inputs.")
        if indexing not in ("ij", "xy"):
            raise ValueError(f"indexing mode {indexing} not supported")

        normalized_inputs = []
        for idx, tensor_input in enumerate(tensor_inputs):
            rank = getattr(tensor_input, "rank", None)
            if rank is not None and rank > 1:
                tensor_input = mb.reshape( # type: ignore
                    x=tensor_input,
                    shape=(-1,),
                    name=f"{node.name}_flatten_{idx}",
                )
            normalized_inputs.append(tensor_input)

        result_symbolic_shape = [tensor_input.shape[0] for tensor_input in normalized_inputs]
        result_shape = ct_ops._utils.maybe_replace_symbols_with_source_tensor_shape_variables(
            result_symbolic_shape,
            normalized_inputs,
        )

        grids = []
        size = len(normalized_inputs)
        for i in range(size):
            view_shape = [1] * size
            view_shape[i] = -1
            view = mb.reshape( # type: ignore
                x=normalized_inputs[i],
                shape=tuple(view_shape),
                name=f"{node.name}_view_{i}",
            )

            reps = result_shape.copy()
            reps[i] = 1
            if any(isinstance(rep, ct_ops.Var) for rep in reps):
                reps = mb.concat(values=reps, axis=0) # type: ignore

            res = mb.tile(x=view, reps=reps, name=f"{node.name}_expand_{i}") # type: ignore

            if indexing == "xy":
                perm = [1, 0] + list(range(2, size))
                res = mb.transpose(x=res, perm=perm, name=f"{node.name}_transpose_{i}") # type: ignore
            grids.append(res)

        context.add(tuple(grids), node.name)

    ct_ops.meshgrid = patched_meshgrid
    _TORCH_OPS_REGISTRY.set_func_by_name(patched_meshgrid, "meshgrid")
    _TORCH_OPS_REGISTRY.set_func_by_name(patched_meshgrid, "meshgrid.indexing")

    def restore():
        ct_ops.meshgrid = original_meshgrid
        if original_meshgrid_reg is not None:
            _TORCH_OPS_REGISTRY.set_func_by_name(original_meshgrid_reg, "meshgrid")
        if original_meshgrid_indexing_reg is not None:
            _TORCH_OPS_REGISTRY.set_func_by_name(
                original_meshgrid_indexing_reg,
                "meshgrid.indexing",
            )

    return restore


def patch_coremltools_split_with_sizes():
    """
    Work around coremltools split converter expecting split_sizes.val.

    Some Torch graphs pass split sizes as Python list/tuple instead of Var.
    """

    import coremltools.converters.mil.frontend.torch.ops as ct_ops
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.frontend.torch.torch_op_registry import _TORCH_OPS_REGISTRY

    original_split = ct_ops.split
    original_split_reg = _TORCH_OPS_REGISTRY.get_func("split")
    original_split_with_sizes_reg = _TORCH_OPS_REGISTRY.get_func("split_with_sizes")
    original_split_with_sizes_copy_reg = _TORCH_OPS_REGISTRY.get_func("split_with_sizes_copy")

    def patched_split(context, node):
        inputs = ct_ops._get_inputs(context, node, min_expected=2)
        nargs = len(inputs)

        x = inputs[0]
        split_sizes = inputs[1]
        dim = inputs[2] if nargs > 2 else 0

        dim = ct_ops._get_kwinputs(context, node, "dim", default=[dim])[0] # type: ignore
        if isinstance(dim, ct_ops.Var):
            dim = dim.val

        if isinstance(split_sizes, (list, tuple)):
            size_tensors = []
            for idx, item in enumerate(split_sizes):
                if isinstance(item, ct_ops.Var):
                    if item.val is not None:
                        size_tensors.append(
                            mb.const( # type: ignore
                                val=[int(item.val)],
                                name=f"{node.name}_split_size_const_{idx}",
                            )
                        )
                    else:
                        rank = getattr(item, "rank", None)
                        tensor_item = item
                        if rank == 0:
                            tensor_item = mb.expand_dims( # type: ignore
                                x=item,
                                axes=[0],
                                name=f"{node.name}_split_size_expand_{idx}",
                            )
                        elif rank is not None and rank > 1:
                            tensor_item = mb.reshape( # type: ignore
                                x=item,
                                shape=(-1,),
                                name=f"{node.name}_split_size_flatten_{idx}",
                            )
                        size_tensors.append(tensor_item)
                else:
                    size_tensors.append(
                        mb.const( # type: ignore
                            val=[int(item)],
                            name=f"{node.name}_split_size_const_{idx}",
                        )
                    )

            if len(size_tensors) == 1:
                split_sizes = size_tensors[0]
            else:
                split_sizes = mb.concat( # type: ignore
                    values=size_tensors,
                    axis=0,
                    name=f"{node.name}_split_sizes_concat",
                )

        if (
            isinstance(split_sizes, ct_ops.Var)
            and getattr(split_sizes, "rank", None) == 0
            and not isinstance(split_sizes.val, np.ndarray)
        ):
            shape = mb.shape(x=x) # type: ignore
            dim_size = ct_ops._list_select(shape, dim)
            num_whole_splits = mb.floor_div(x=dim_size, y=split_sizes) # type: ignore
            remainder = mb.mod(x=dim_size, y=split_sizes) # type: ignore

            tmp = mb.const(val=[1]) # type: ignore
            whole_sizes = mb.mul(x=tmp, y=split_sizes) # type: ignore
            reps = mb.mul(x=tmp, y=num_whole_splits) # type: ignore
            whole_sizes = mb.tile(x=whole_sizes, reps=reps) # type: ignore
            if remainder.val == 0:
                split_sizes = whole_sizes
            else:
                partial_size = mb.mul(x=tmp, y=remainder) # type: ignore
                split_sizes = mb.concat(values=[whole_sizes, partial_size], axis=0) # type: ignore

        res = mb.split(x=x, split_sizes=split_sizes, axis=dim, name=node.name) # type: ignore
        context.add(res, torch_name=node.name)

    ct_ops.split = patched_split
    _TORCH_OPS_REGISTRY.set_func_by_name(patched_split, "split")
    _TORCH_OPS_REGISTRY.set_func_by_name(patched_split, "split_with_sizes")
    _TORCH_OPS_REGISTRY.set_func_by_name(patched_split, "split_with_sizes_copy")

    def restore():
        ct_ops.split = original_split
        if original_split_reg is not None:
            _TORCH_OPS_REGISTRY.set_func_by_name(original_split_reg, "split")
        if original_split_with_sizes_reg is not None:
            _TORCH_OPS_REGISTRY.set_func_by_name(
                original_split_with_sizes_reg,
                "split_with_sizes",
            )
        if original_split_with_sizes_copy_reg is not None:
            _TORCH_OPS_REGISTRY.set_func_by_name(
                original_split_with_sizes_copy_reg,
                "split_with_sizes_copy",
            )

    return restore


class RFDETRInferenceWrapper(nn.Module):
    """
    Wrapper that returns (boxes, scores) for CoreML conversion.
    """

    def __init__(self, model: nn.Module, call_mode: str, values_are_logits: bool):
        super().__init__()
        self.model = model
        self.call_mode = call_mode
        self.values_are_logits = values_are_logits

    def forward(self, images: torch.Tensor):
        out = _run_with_mode(self.model, images, self.call_mode)
        boxes, values, _ = _resolve_output_tensors(out)

        if boxes.ndim == 2:
            boxes = boxes.unsqueeze(0)
        if values.ndim == 2:
            values = values.unsqueeze(0)

        scores = torch.sigmoid(values) if self.values_are_logits else values
        return boxes, scores


def export_torch_program(
    wrapper: nn.Module,
    input_size: int,
    device: str,
    export_backend: str,
):
    dummy_input = torch.zeros(1, 3, input_size, input_size, device=device)

    with torch.no_grad():
        def _export_torchscript():
            traced_local = torch.jit.trace(wrapper, dummy_input, strict=False)
            print("Export backend : torch.jit.trace")
            print("boxes shape    : resolved at CoreML conversion time")
            print("scores shape   : resolved at CoreML conversion time")
            return traced_local

        def _export_torch_export():
            exported_local = torch.export.export(wrapper, (dummy_input,), strict=False)
            exported_local = exported_local.run_decompositions({})
            boxes_local, scores_local = exported_local.module()(dummy_input)
            print("Export backend : torch.export")
            print(f"boxes shape    : {tuple(boxes_local.shape)}")
            print(f"scores shape   : {tuple(scores_local.shape)}")
            return exported_local

        if export_backend == "torchscript":
            return _export_torchscript()

        if export_backend == "torch_export":
            return _export_torch_export()

        # auto: prefer torch.export first, then fall back to torchscript.
        try:
            return _export_torch_export()
        except Exception as exc:
            print(f"torch.export failed, falling back to torch.jit.trace: {exc}")
            return _export_torchscript()


def convert_to_coreml(
    torch_program: Any,
    input_size: int,
    output_path: Path,
    compute_precision: str,
    compute_units: str,
    short_description: str,
):
    image_input = ct.ImageType(
        name="image",
        shape=(1, 3, input_size, input_size),
        scale=1.0 / 255.0,
        color_layout=ct.colorlayout.RGB,
    )

    restore_inplace_copy = patch_coremltools_tensor_inplace_copy()
    restore_meshgrid = patch_coremltools_meshgrid()
    restore_split = patch_coremltools_split_with_sizes()
    try:
        mlmodel: Any = ct.convert(
            torch_program,
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.macOS13,
            inputs=[image_input],
            outputs=[ct.TensorType(name="boxes"), ct.TensorType(name="scores")],
            compute_precision=(
                ct.precision.FLOAT16 if compute_precision == "float16" else ct.precision.FLOAT32
            ),
            compute_units=COMPUTE_UNIT_MAP[compute_units],
        )
    finally:
        restore_inplace_copy()
        restore_meshgrid()
        restore_split()

    mlmodel.short_description = short_description
    mlmodel.input_description["image"] = (
        f"RGB image, {input_size}x{input_size}, pixel values 0-255"
    )
    mlmodel.output_description["boxes"] = (
        "Bounding boxes [1, Q, 4] as (cx, cy, w, h) normalized 0-1"
    )
    mlmodel.output_description["scores"] = (
        "Class probabilities [1, Q, C] after sigmoid"
    )
    mlmodel.author = "CodePerfectplus | Deepak Raj"
    mlmodel.version = "1.0"

    mlmodel.save(str(output_path))

    if output_path.is_file():
        size_mb = output_path.stat().st_size / 1e6
    else:
        size_mb = sum(p.stat().st_size for p in output_path.rglob("*") if p.is_file()) / 1e6
    print(f"Saved CoreML model: {output_path} ({size_mb:.1f} MB)")

    return mlmodel


def _is_ane_compile_error(exc: Exception) -> bool:
    msg = str(exc)
    return (
        "MILCompilerForANE" in msg
        or "ANECCompile() FAILED" in msg
        or "_ANECompiler" in msg
        or "failed to compile ANE model" in msg
    )


def benchmark(mlmodel, input_size: int = 640, n_runs: int = 50):
    import statistics
    import time

    dummy = np.random.randint(0, 255, (input_size, input_size, 3), dtype=np.uint8)
    pil_img = Image.fromarray(dummy)

    latencies = []
    for i in range(n_runs + 5):
        t0 = time.perf_counter()
        _ = mlmodel.predict({"image": pil_img})
        t1 = time.perf_counter()
        if i >= 5:
            latencies.append((t1 - t0) * 1000)

    med = statistics.median(latencies)
    print(f"\nBenchmark ({n_runs} runs, {input_size}^2):")
    print(f"  Median latency : {med:.1f} ms  ->  ~{1000 / med:.1f} FPS")
    print(f"  Min / Max      : {min(latencies):.1f} / {max(latencies):.1f} ms")
    return {
        "median_ms": med,
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "fps": 1000.0 / med,
    }


def benchmark_compute_units(
    model_path: str,
    input_size: int,
    n_runs: int,
    compute_units_order,
):
    print("\n=== Runtime Compute Unit Benchmark ===")
    print("This benchmarks the saved .mlpackage with different runtime backends.")

    results = []
    for cu in compute_units_order:
        print(f"\nTrying runtime compute_units={cu}...")
        try:
            runtime_model = ct.models.MLModel(
                model_path,
                compute_units=getattr(ct.ComputeUnit, cu),
            )
            stats = benchmark(runtime_model, input_size=input_size, n_runs=n_runs)
            stats["compute_units"] = cu
            results.append(stats)
        except Exception as exc:
            if _is_ane_compile_error(exc):
                print(f"  Skipping {cu}: ANE compile failed ({exc})")
                continue
            raise

    if not results:
        raise RuntimeError(
            "All benchmark runtime compute unit choices failed. "
            "Try --benchmark_compute_units CPU_AND_GPU,CPU_ONLY"
        )

    best = max(results, key=lambda x: x["fps"])
    print("\nBest runtime backend:")
    print(
        f"  {best['compute_units']} -> {best['median_ms']:.1f} ms median "
        f"(~{best['fps']:.1f} FPS)"
    )

    return results, best


def main():
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_path = Path(args.output) if args.output else checkpoint_path.with_suffix(".mlpackage")

    constructor_kwargs = json.loads(args.constructor_kwargs)
    if not isinstance(constructor_kwargs, dict):
        raise ValueError("--constructor_kwargs must be a JSON object")

    effective_input_size = resolve_effective_input_size(
        requested_size=args.input_size,
        backbone_stride=args.backbone_stride,
        no_auto_adjust_input_size=args.no_auto_adjust_input_size,
    )

    print("=== RF-DETR -> CoreML Conversion ===")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Input size : {args.input_size}x{args.input_size}")
    print(f"Precision  : {args.compute_precision}")
    print(f"Compute    : {args.compute_units}")
    print(f"Backend    : {args.export_backend}")

    state_dict, metadata = checkpoint_to_state_dict(checkpoint_path)
    inferred_num_classes = infer_num_classes_from_state_dict(state_dict)
    num_classes = args.num_classes if args.num_classes > 0 else inferred_num_classes

    print(f"Checkpoint       : {checkpoint_path}")
    print(f"State dict keys  : {len(state_dict)}")
    if metadata.get("checkpoint_keys"):
        print(f"Top-level keys   : {metadata['checkpoint_keys']}")
    print(f"Requested size   : {args.input_size}")
    print(f"Effective size   : {effective_input_size}")
    print(f"Inferred classes : {inferred_num_classes}")
    print(f"Using classes    : {num_classes}")

    model_obj, used_kwargs = instantiate_model_object(
        args.model_class,
        num_classes,
        constructor_kwargs,
    )
    print(f"Model class      : {args.model_class}")
    print(f"Ctor kwargs      : {used_kwargs}")

    model = extract_torch_module(model_obj)
    model = model.to(args.device)
    model.eval()

    load_msg, dropped_mismatch = load_weights_flexible(model, state_dict, strict=args.strict_load)

    if dropped_mismatch:
        print("Dropped shape-mismatched tensors:")
        for key, ckpt_shape, model_shape in dropped_mismatch[:20]:
            print(f"  {key}: ckpt{ckpt_shape} != model{model_shape}")
        if len(dropped_mismatch) > 20:
            print(f"  ... and {len(dropped_mismatch) - 20} more")

    missing = list(getattr(load_msg, "missing_keys", None) or [])
    unexpected = list(getattr(load_msg, "unexpected_keys", None) or [])
    print(f"Missing keys      : {len(missing)}")
    print(f"Unexpected keys   : {len(unexpected)}")

    if not args.no_mha_patch:
        replace_multihead_attention_for_export(model)
        print("Applied MHA patch : yes")
    else:
        print("Applied MHA patch : no")

    if not args.no_msda_patch:
        replace_ms_deform_attn_for_export(model)
        print("Applied MSDA patch: yes")
    else:
        print("Applied MSDA patch: no")

    restore_interpolate = patch_interpolate_for_export(args.no_bicubic_patch)
    if args.no_bicubic_patch:
        print("Bicubic patch     : disabled")
    else:
        print("Bicubic patch     : enabled (bicubic->bilinear, antialias off)")

    try:
        call_mode, values_are_logits, boxes_shape, values_shape = detect_forward_mode_and_output(
            model,
            effective_input_size,
            args.device,
        )
        print(f"Call mode         : {call_mode}")
        print(f"Values are logits : {values_are_logits}")
        print(f"Probe boxes shape : {boxes_shape}")
        print(f"Probe value shape : {values_shape}")

        wrapper = RFDETRInferenceWrapper(model, call_mode=call_mode, values_are_logits=values_are_logits)
        wrapper = wrapper.to(args.device)
        wrapper.eval()

        torch_program = export_torch_program(
            wrapper,
            effective_input_size,
            args.device,
            args.export_backend,
        )
    finally:
        restore_interpolate()

    short_desc = f"RF-DETR real-time object detector ({args.model_class})"
    mlmodel = convert_to_coreml(
        torch_program=torch_program,
        input_size=effective_input_size,
        output_path=output_path,
        compute_precision=args.compute_precision,
        compute_units=args.compute_units,
        short_description=short_desc,
    )

    if args.verify:
        _ = ct.models.MLModel(str(output_path), compute_units=COMPUTE_UNIT_MAP[args.compute_units])
        print("Verification      : loaded successfully")

    if args.benchmark:
        cu_list = [x.strip() for x in args.benchmark_compute_units.split(",") if x.strip()]
        valid = {"ALL", "CPU_AND_NE", "CPU_AND_GPU", "CPU_ONLY"}
        invalid = [x for x in cu_list if x not in valid]
        if invalid:
            raise ValueError(
                f"Invalid --benchmark_compute_units values: {invalid}. "
                f"Valid values: {sorted(valid)}"
            )

        benchmark_compute_units(
            model_path=str(output_path),
            input_size=effective_input_size,
            n_runs=args.benchmark_runs,
            compute_units_order=cu_list,
        )

    # Keep a reference alive for some toolchains.
    _ = mlmodel


if __name__ == "__main__":
    main()
