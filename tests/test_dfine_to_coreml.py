from pathlib import Path

import pytest

ct = pytest.importorskip("coremltools")
torch = pytest.importorskip("torch")
import torch.nn as nn

import export_dfine_coreml as converter


class _DummyDetector(nn.Module):
    def forward(self, _images):
        boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.2]]], dtype=torch.float32)
        logits = torch.tensor([[[0.0, 2.0, -2.0]]], dtype=torch.float32)
        return {"pred_boxes": boxes, "pred_logits": logits}


def test_dfine_inference_wrapper_returns_sigmoid_scores():
    wrapper = converter.DFineInferenceWrapper(_DummyDetector())

    boxes, scores = wrapper(torch.zeros(1, 3, 8, 8, dtype=torch.float32))

    assert boxes.shape == (1, 1, 4)
    assert scores.shape == (1, 1, 3)
    assert scores[0, 0, 1].item() == pytest.approx(torch.sigmoid(torch.tensor(2.0)).item())


def test_coreml_friendly_attention_matches_pytorch_attention_output():
    mha = nn.MultiheadAttention(embed_dim=8, num_heads=2, dropout=0.0, batch_first=True)
    wrapper = converter.CoreMLFriendlyMultiheadAttention(mha)

    query = torch.randn(2, 5, 8)
    key = torch.randn(2, 5, 8)
    value = torch.randn(2, 5, 8)

    ref_out, _ = mha(query, key, value, need_weights=False)
    wrapped_out, wrapped_weights = wrapper(query, key, value, need_weights=False)

    assert wrapped_weights is None
    assert wrapped_out.shape == ref_out.shape
    assert torch.allclose(wrapped_out, ref_out, atol=1e-5, rtol=1e-4)


def test_replace_multihead_attention_for_export_replaces_nested_modules():
    class _Nested(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.MultiheadAttention(embed_dim=8, num_heads=2, batch_first=True)
            self.inner = nn.Module()
            self.inner.attn = nn.MultiheadAttention(embed_dim=8, num_heads=2, batch_first=True)

    model = _Nested()
    converter.replace_multihead_attention_for_export(model)

    assert isinstance(model.attn, converter.CoreMLFriendlyMultiheadAttention)
    assert isinstance(model.inner.attn, converter.CoreMLFriendlyMultiheadAttention)


def test_prepare_frontend_model_auto_falls_back_to_torchscript(monkeypatch):
    expected = object()

    def fake_export_model(*_args, **_kwargs):
        raise RuntimeError("export failed")

    def fake_trace_model(*_args, **_kwargs):
        return expected

    monkeypatch.setattr(converter, "export_model", fake_export_model)
    monkeypatch.setattr(converter, "trace_model", fake_trace_model)

    model = nn.Identity()
    frontend_model, used_frontend = converter.prepare_frontend_model(
        model,
        input_size=320,
        device="cpu",
        frontend="auto",
    )

    assert frontend_model is expected
    assert used_frontend == "torchscript"


def test_convert_to_coreml_uses_requested_precision_and_saves_package(monkeypatch, tmp_path):
    captured = {}

    class _FakeMLModel:
        def __init__(self):
            self.short_description = ""
            self.input_description = {}
            self.output_description = {}
            self.saved_path = None

        def save(self, path):
            pkg = Path(path)
            pkg.mkdir(parents=True, exist_ok=True)
            (pkg / "weights.bin").write_bytes(b"x" * 256)
            self.saved_path = pkg

    fake_model = _FakeMLModel()

    def fake_convert(torch_model, **kwargs):
        captured["torch_model"] = torch_model
        captured.update(kwargs)
        return fake_model

    monkeypatch.setattr(converter.ct, "convert", fake_convert)

    output_path = tmp_path / "fake_model.mlpackage"
    result = converter.convert_to_coreml(
        torch_model="dummy_graph",
        input_size=320,
        output_path=str(output_path),
        compute_precision="float16",
    )

    assert result is fake_model
    assert output_path.exists() and output_path.is_dir()
    assert captured["convert_to"] == "mlprogram"
    assert captured["compute_precision"] == converter.ct.precision.FLOAT16
    assert captured["compute_units"] == converter.ct.ComputeUnit.CPU_AND_GPU
    assert fake_model.saved_path == output_path


def test_is_ane_compile_error_detects_known_patterns():
    assert converter._is_ane_compile_error(RuntimeError("ANECCompile() FAILED"))
    assert converter._is_ane_compile_error(RuntimeError("MILCompilerForANE error"))
    assert not converter._is_ane_compile_error(RuntimeError("random runtime error"))
