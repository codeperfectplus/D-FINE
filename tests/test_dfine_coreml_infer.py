import types

import numpy as np
import pytest
from PIL import Image

import coremltools as ct

from dfine_coreml_infer import DFineCoreMLPredictor


def _build_stub_predictor(conf_threshold: float = 0.4):
    predictor = DFineCoreMLPredictor.__new__(DFineCoreMLPredictor)
    predictor.conf_threshold = conf_threshold
    predictor.class_names = ["person", "car", "dog"]
    predictor.input_width = 100
    predictor.input_height = 100
    predictor.compute_units = ct.ComputeUnit.ALL
    predictor.output_format = "dfine"
    predictor.primary_output_key = "boxes"
    return predictor


def test_infer_output_format_detects_dfine_and_yolo():
    predictor = _build_stub_predictor()

    fmt, key = predictor._infer_output_format(
        {
            "boxes": np.zeros((1, 2, 4), dtype=np.float32),
            "scores": np.zeros((1, 2, 3), dtype=np.float32),
        }
    )
    assert (fmt, key) == ("dfine", "boxes")

    fmt, key = predictor._infer_output_format(
        {
            "prediction": np.zeros((1, 10, 6), dtype=np.float32),
        }
    )
    assert (fmt, key) == ("yolo_nms6", "prediction")


@pytest.mark.parametrize(
    "bad_output",
    [
        None,
        {"foo": np.zeros((1, 2, 3), dtype=np.float32)},
    ],
)
def test_infer_output_format_raises_for_unsupported_output(bad_output):
    predictor = _build_stub_predictor()
    with pytest.raises(RuntimeError):
        predictor._infer_output_format(bad_output)


def test_decode_filters_threshold_and_scales_boxes():
    predictor = _build_stub_predictor(conf_threshold=0.4)

    boxes = np.array(
        [
            [0.5, 0.5, 0.4, 0.4],
            [0.2, 0.2, 0.2, 0.2],
        ],
        dtype=np.float32,
    )
    scores = np.array(
        [
            [0.1, 0.9, 0.0],
            [0.2, 0.39, 0.1],
        ],
        dtype=np.float32,
    )

    detections = predictor._decode(boxes, scores, orig_h=100, orig_w=200)

    assert len(detections) == 1
    det = detections[0]
    assert det["class_id"] == 1
    assert det["class_name"] == "car"
    assert det["box_xyxy"] == pytest.approx([60.0, 30.0, 140.0, 70.0], abs=1e-4)


def test_decode_yolo_nms6_scales_coordinates():
    predictor = _build_stub_predictor(conf_threshold=0.3)
    predictor.input_width = 100
    predictor.input_height = 100

    predictions = np.array(
        [
            [10.0, 10.0, 40.0, 40.0, 0.95, 2.0],
            [0.0, 0.0, 50.0, 50.0, 0.2, 0.0],
        ],
        dtype=np.float32,
    )

    detections = predictor._decode_yolo_nms6(predictions, orig_h=200, orig_w=200)

    assert len(detections) == 1
    det = detections[0]
    assert det["class_id"] == 2
    assert det["class_name"] == "dog"
    assert det["box_xyxy"] == pytest.approx([20.0, 20.0, 80.0, 80.0], abs=1e-4)


def test_predict_falls_back_to_cpu_and_gpu_on_ane_compile_error():
    predictor = _build_stub_predictor(conf_threshold=0.1)

    output = {
        "boxes": np.array([[[0.5, 0.5, 0.2, 0.2]]], dtype=np.float32),
        "scores": np.array([[[0.95, 0.02, 0.03]]], dtype=np.float32),
    }

    class FailingModel:
        def predict(self, _payload):
            raise RuntimeError("MILCompilerForANE: ANECCompile() FAILED")

    class FallbackModel:
        def __init__(self, result):
            self.result = result
            self.calls = 0

        def predict(self, _payload):
            self.calls += 1
            return self.result

    fallback_model = FallbackModel(output)
    loaded = {}

    def fake_load_model(self, compute_units=None):
        loaded["compute_units"] = compute_units
        return fallback_model

    def fake_load_image(self, _source):
        image = Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))
        return image, (10, 10)

    predictor.model = FailingModel()
    predictor._load_model = types.MethodType(fake_load_model, predictor)
    predictor._load_image = types.MethodType(fake_load_image, predictor)

    detections, latency_ms = predictor.predict(np.zeros((10, 10, 3), dtype=np.uint8))

    assert predictor.compute_units == ct.ComputeUnit.CPU_AND_GPU
    assert loaded["compute_units"] == ct.ComputeUnit.CPU_AND_GPU
    assert fallback_model.calls == 1
    assert len(detections) == 1
    assert latency_ms >= 0
