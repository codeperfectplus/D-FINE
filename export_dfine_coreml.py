"""
D-FINE → CoreML Conversion Script
===================================
Converts D-FINE (ICLR 2025) PyTorch model to CoreML .mlpackage
for ANE/GPU-accelerated inference on Apple Silicon (M1/M2/M3/M4).

Requirements:
    pip install coremltools>=8.0 torch torchvision

Usage:
    python dfine_to_coreml.py \
        --config configs/dfine/dfine_hgnetv2_l_coco.yml \
        --checkpoint dfine_l_coco.pth \
        --size 640 \
        --output dfine_l.mlpackage
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import coremltools as ct
from pathlib import Path
from typing import Any, Tuple
from PIL import Image

from transformers import AutoModelForObjectDetection


from src.core import YAMLConfig
# ─────────────────────────────────────────────────────────────────
# 1. Inference Wrapper
#    D-FINE's forward() returns a dict with aux losses etc.
#    CoreML needs a clean (boxes, scores) tuple output.
#    We also bake in sigmoid on logits so the .mlpackage is
#    self-contained (no post-processing needed in Swift/Python).
# ─────────────────────────────────────────────────────────────────
class DFineInferenceWrapper(nn.Module):
    """
    Wraps D-FINE so that:
      - Input  : float32 image tensor [1, 3, H, W], range [0, 1]
      - Outputs: boxes  [1, num_queries, 4]  (cx, cy, w, h), normalized 0-1
                 scores [1, num_queries, num_classes]  (sigmoid probabilities)
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor):
        # D-FINE expects images in [0,1] by default (it normalizes internally).
        # If your model was trained with a custom normalizer, apply it here.
        outputs = self.model(images)

        # D-FINE returns a dict; keys may vary slightly by version.
        # Common keys: 'pred_boxes', 'pred_logits'
        boxes  = outputs["pred_boxes"]   # [B, Q, 4]  cx/cy/w/h normalized
        logits = outputs["pred_logits"]  # [B, Q, C]
        scores = torch.sigmoid(logits)   # convert logits → probabilities

        return boxes, scores


class CoreMLFriendlyMultiheadAttention(nn.Module):
    """
    Export-friendly drop-in replacement for nn.MultiheadAttention (batch_first=True).
    It keeps original weights but uses explicit linear projections + SDPA.
    """

    def __init__(self, mha: nn.MultiheadAttention):
        super().__init__()
        if not mha.batch_first:
            raise ValueError("Only batch_first=True MultiheadAttention is supported")
        if mha.kdim is not None and mha.kdim != mha.embed_dim:
            raise ValueError("kdim must match embed_dim for export replacement")
        if mha.vdim is not None and mha.vdim != mha.embed_dim:
            raise ValueError("vdim must match embed_dim for export replacement")

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

    def _project_qkv(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        return q, k, v

    def _reshape_heads(self, x: torch.Tensor):
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
        # key_padding_mask is unused in D-FINE inference path.
        if key_padding_mask is not None:
            raise ValueError("key_padding_mask is not supported in export-friendly attention")

        q, k, v = self._project_qkv(query, key, value)
        q = self._reshape_heads(q)
        k = self._reshape_heads(k)
        v = self._reshape_heads(v)

        attn_mask_t = None
        if attn_mask is not None:
            # Accept common 2D mask [L, S]; keep bool/float type as-is.
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
        attn_output = attn_output.transpose(1, 2).reshape(query.shape[0], query.shape[1], self.embed_dim)
        attn_output = self.out_proj(attn_output)

        if not need_weights:
            return attn_output, None

        # Weights are unused by conversion path, so return placeholder with correct rank.
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


def replace_multihead_attention_for_export(module: nn.Module):
    """Recursively replace nn.MultiheadAttention with export-friendly equivalent."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.MultiheadAttention):
            setattr(module, name, CoreMLFriendlyMultiheadAttention(child))
        else:
            replace_multihead_attention_for_export(child)


# ─────────────────────────────────────────────────────────────────
# 2. Load D-FINE from the official repo
# ─────────────────────────────────────────────────────────────────
def load_dfine(
    config_path: str,
    checkpoint_path: str,
    device: str = "mps",
    input_size: int = 640,
):
    """
    Loads D-FINE using the official Peterande/D-FINE codebase.
    The D-FINE repo must be on sys.path (run from its root dir, or add it).
    """
    cfg = YAMLConfig(config_path, resume=checkpoint_path)

    # Keep eval spatial size aligned with export resolution so positional
    # embeddings and anchor grids are generated for the requested input size.
    eval_spatial_size = [int(input_size), int(input_size)]
    cfg.yaml_cfg["eval_spatial_size"] = eval_spatial_size
    if "HybridEncoder" in cfg.yaml_cfg:
        cfg.yaml_cfg["HybridEncoder"]["eval_spatial_size"] = eval_spatial_size
    if "DFINETransformer" in cfg.yaml_cfg:
        cfg.yaml_cfg["DFINETransformer"]["eval_spatial_size"] = eval_spatial_size

    # Avoid downloading backbone pretrain weights when checkpoint is provided.
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    checkpoint = torch.load(checkpoint_path, map_location="mps")
    if "ema" in checkpoint:
        state_dict = checkpoint["ema"]["module"]
    else:
        state_dict = checkpoint["model"]

    model = cfg.model

    # Checkpoints may contain resolution-dependent buffers (for example,
    # decoder anchors/valid masks). Skip only incompatible tensor shapes so
    # exporting at different input sizes can still reuse trained weights.
    model_state = model.state_dict()
    filtered_state_dict = {}
    dropped_mismatch = []
    for key, value in state_dict.items():
        if key in model_state and model_state[key].shape != value.shape:
            dropped_mismatch.append((key, tuple(value.shape), tuple(model_state[key].shape)))
            continue
        filtered_state_dict[key] = value

    if dropped_mismatch:
        print("Skipped shape-mismatched checkpoint tensors:")
        for key, ckpt_shape, model_shape in dropped_mismatch:
            print(f"  {key}: ckpt{ckpt_shape} != model{model_shape}")

    load_msg = model.load_state_dict(filtered_state_dict, strict=False)

    missing_keys = list(getattr(load_msg, "missing_keys", None) or [])
    optional_missing = {"encoder.pos_embed2", "decoder.anchors", "decoder.valid_mask"}
    non_optional_missing = [k for k in missing_keys if k not in optional_missing]
    if non_optional_missing:
        print(f"Missing keys while loading checkpoint (ignored): {non_optional_missing}")
    elif missing_keys:
        print(f"Missing optional buffers (safe to ignore): {missing_keys}")

    if getattr(load_msg, "unexpected_keys", None):
        print(f"Unexpected keys while loading checkpoint (ignored): {load_msg.unexpected_keys}")

    # Convert train-time modules to deploy/inference mode where available.
    if hasattr(model, "deploy"):
        model = model.deploy() # type: ignore

    # Replace built-in MHA to avoid known CoreML conversion failures in attention ops.
    replace_multihead_attention_for_export(model)
    

    model.eval()
    model = model.to(device)
    return model


# ─────────────────────────────────────────────────────────────────
# 3. Trace the model
#    D-FINE uses deformable attention with dynamic shapes internally.
#    Tracing at a FIXED resolution side-steps those issues.
#    Use 640×640 for COCO models; 320×320 for speed if acceptable.
# ─────────────────────────────────────────────────────────────────
def trace_model(
    model: nn.Module,
    input_size: int = 640,
    device: str = "mps",
) -> torch.jit.ScriptModule:
    wrapper = DFineInferenceWrapper(model).to(device)
    wrapper.eval()

    dummy_input = torch.zeros(1, 3, input_size, input_size, device=device)

    print(f"Tracing model at {input_size}×{input_size}…")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, dummy_input, strict=False)

    # Quick sanity check
    with torch.no_grad():
        boxes, scores = traced(dummy_input) # type: ignore
    print(f"  boxes  shape: {boxes.shape}")
    print(f"  scores shape: {scores.shape}")
    return traced # type: ignore


def export_model(
    model: nn.Module,
    input_size: int = 640,
    device: str = "mps",
):
    """
    Export model via torch.export (ExportedProgram), which is often
    more robust than TorchScript tracing for CoreML conversion.
    """
    wrapper = DFineInferenceWrapper(model).to(device)
    wrapper.eval()

    dummy_input = torch.zeros(1, 3, input_size, input_size, device=device)

    print(f"Exporting model with torch.export at {input_size}×{input_size}…")
    with torch.no_grad():
        exported = torch.export.export(wrapper, (dummy_input,), strict=False)
        # CoreML conversion expects decomposed ATEN/EDGE graph instead of TRAINING dialect.
        exported = exported.run_decompositions({})

    # Quick sanity check from exported graph module
    with torch.no_grad():
        gm = exported.module()
        boxes, scores = gm(dummy_input)
    print(f"  boxes  shape: {boxes.shape}")
    print(f"  scores shape: {scores.shape}")
    return exported


# ─────────────────────────────────────────────────────────────────
# 4. Convert to CoreML
# ─────────────────────────────────────────────────────────────────
def convert_to_coreml(
    torch_model: Any,
    input_size: int = 640,
    output_path: str = "dfine.mlpackage",
    compute_precision: str = "float16",   # "float16" or "float32"
):
    """
    Convert traced D-FINE to CoreML .mlpackage.
    Uses mlprogram format (iOS16+/macOS13+) for ANE support.
    """

    shape = (1, 3, input_size, input_size)

    # Use ImageType so Swift/ObjC callers can pass CVPixelBuffer directly.
    # scale=1/255 normalizes [0,255] → [0,1].
    image_input = ct.ImageType(
        name="image",
        shape=shape,
        scale=1.0 / 255.0,
        color_layout=ct.colorlayout.RGB,
    )

    print("Converting to CoreML (mlprogram format)…")
    mlmodel = ct.convert(
        torch_model,
        inputs=[image_input],
        outputs=[
            ct.TensorType(name="boxes"),
            ct.TensorType(name="scores"),
        ],
        convert_to="mlprogram",          # required for ANE offload
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=(
            ct.precision.FLOAT16
            if compute_precision == "float16"
            else ct.precision.FLOAT32
        ),
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
    )

    # ── Metadata ───────────────────────────────────────────────────
    mlmodel.short_description = "D-FINE real-time object detector (ICLR 2025)" # type: ignore
    mlmodel.input_description["image"] = ( # type: ignore
        f"RGB image, {input_size}×{input_size}, pixel values 0–255"
    )
    mlmodel.output_description["boxes"] = ( # type: ignore
        "Bounding boxes [1, Q, 4] as (cx, cy, w, h) normalized 0–1"
    )
    mlmodel.output_description["scores"] = ( # type: ignore
        "Class probabilities [1, Q, C] after sigmoid"
    )
    mlmodel.author = "Scry AI" # type: ignore
    mlmodel.version = "1.0" # type: ignore


    mlmodel.save(output_path) # type: ignore
    out_path = Path(output_path)
    size_mb = None
    if out_path.is_file():
        size_mb = out_path.stat().st_size / 1e6
    elif out_path.is_dir():
        size_mb = sum(p.stat().st_size for p in out_path.rglob("*") if p.is_file()) / 1e6

    if size_mb is None:
        print(f"Saved: {output_path}")
    else:
        print(f"Saved: {output_path}  ({size_mb:.1f} MB)")
    return mlmodel


# ─────────────────────────────────────────────────────────────────
# 5. Benchmark (Python)
# ─────────────────────────────────────────────────────────────────
def _is_ane_compile_error(exc: Exception) -> bool:
    msg = str(exc)
    return (
        "MILCompilerForANE" in msg
        or "ANECCompile() FAILED" in msg
        or "_ANECompiler" in msg
        or "failed to compile ANE model" in msg
    )


def benchmark(mlmodel, input_size: int = 640, n_runs: int = 50):
    import time, statistics

    dummy = np.random.randint(0, 255, (input_size, input_size, 3), dtype=np.uint8)

    pil_img = Image.fromarray(dummy)

    latencies = []
    for i in range(n_runs + 5):          # 5-run warm-up
        t0 = time.perf_counter()
        _ = mlmodel.predict({"image": pil_img})
        t1 = time.perf_counter()
        if i >= 5:
            latencies.append((t1 - t0) * 1000)

    med = statistics.median(latencies)
    print(f"\nBenchmark ({n_runs} runs, {input_size}²):")
    print(f"  Median latency : {med:.1f} ms  →  ~{1000/med:.1f} FPS")
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
        print(f"\nTrying runtime compute_units={cu}…")
        try:
            runtime_model = ct.models.MLModel(
                model_path,
                compute_units=getattr(ct.ComputeUnit, cu),
            )
            stats = benchmark(runtime_model, input_size=input_size, n_runs=n_runs)
            stats["compute_units"] = cu
            results.append(stats)
        except Exception as e:
            if _is_ane_compile_error(e):
                print(f"  Skipping {cu}: ANE compile failed ({e})")
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


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Convert D-FINE to CoreML")
    p.add_argument("--config",     required=True,  help="Path to D-FINE YAML config")
    p.add_argument("--checkpoint", required=True,  help="Path to .pth checkpoint")
    p.add_argument("--size",       type=int, default=640, help="Input resolution (square)")
    p.add_argument("--output",     default="dfine.mlpackage")
    p.add_argument("--precision",  choices=["float16", "float32"], default="float16")
    p.add_argument(
        "--frontend",
        choices=["auto", "torch_export", "torchscript"],
        default="auto",
        help=(
            "PyTorch frontend for CoreML conversion. "
            "'auto' tries torch_export first then falls back to torchscript."
        ),
    )
    p.add_argument("--benchmark",  action="store_true", help="Run latency benchmark after conversion")
    p.add_argument(
        "--benchmark_runs",
        type=int,
        default=50,
        help="Number of timed runs for each benchmark backend",
    )
    p.add_argument(
        "--benchmark_compute_units",
        default="ALL,CPU_AND_GPU,CPU_AND_NE,CPU_ONLY",
        help=(
            "Comma-separated runtime compute units to benchmark against the saved model. "
            "Order matters. Example: CPU_AND_GPU,CPU_ONLY"
        ),
    )
    p.add_argument("--device",     default="mps", help="Device for tracing (cpu recommended)")
    return p.parse_args()


def prepare_frontend_model(
    model: nn.Module,
    input_size: int,
    device: str,
    frontend: str,
) -> Tuple[Any, str]:
    if frontend == "torchscript":
        return trace_model(model, input_size=input_size, device=device), "torchscript"

    if frontend == "torch_export":
        return export_model(model, input_size=input_size, device=device), "torch_export"

    # auto mode
    try:
        torch_model = export_model(model, input_size=input_size, device=device)
        return torch_model, "torch_export"
    except Exception as e:
        print(f"torch.export failed: {e}")
        print("Falling back to TorchScript trace frontend…")
        torch_model = trace_model(model, input_size=input_size, device=device)
        return torch_model, "torchscript"


def main():
    args = parse_args()

    print("=== D-FINE → CoreML Conversion ===")
    print(f"Config     : {args.config}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Input size : {args.size}×{args.size}")
    print(f"Precision  : {args.precision}")
    print("Compute    : CPU_AND_GPU (fixed)")
    print(f"Frontend   : {args.frontend}")

    model  = load_dfine(
        args.config,
        args.checkpoint,
        device=args.device,
        input_size=args.size,
    )
    torch_model, used_frontend = prepare_frontend_model(
        model,
        input_size=args.size,
        device=args.device,
        frontend=args.frontend,
    )
    print(f"Using frontend: {used_frontend}")

    try:
        mlmodel = convert_to_coreml(
            torch_model,
            input_size=args.size,
            output_path=args.output,
            compute_precision=args.precision,
        )
    except Exception as e:
        if args.frontend == "auto" and used_frontend == "torch_export":
            print(f"torch_export conversion failed: {e}")
            print("Retrying conversion with TorchScript frontend…")
            torch_model = trace_model(model, input_size=args.size, device=args.device)
            used_frontend = "torchscript"
            print(f"Using frontend: {used_frontend}")
            mlmodel = convert_to_coreml(
                torch_model,
                input_size=args.size,
                output_path=args.output,
                compute_precision=args.precision,
            )
        else:
            raise

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
            model_path=args.output,
            input_size=args.size,
            n_runs=args.benchmark_runs,
            compute_units_order=cu_list,
        )


if __name__ == "__main__":
    main()