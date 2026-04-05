"""
D-FINE CoreML Inference Helper
================================
Drop-in replacement for the MPS/CUDA PyTorch inference loop.
Runs the converted .mlpackage on Apple Neural Engine + GPU.

Usage:
    from dfine_coreml_infer import DFineCoreMLPredictor

    predictor = DFineCoreMLPredictor("dfine_l.mlpackage", conf_threshold=0.4)
    results = predictor.predict("image.jpg")
    predictor.draw(results, "image.jpg", "output.jpg")
"""

import numpy as np
from PIL import Image
import coremltools as ct
import time
from pathlib import Path


COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
    "book","clock","vase","scissors","teddy bear","hair drier","toothbrush",
]


class DFineCoreMLPredictor:
    """
    Wraps a D-FINE .mlpackage for easy Python inference.

    Parameters
    ----------
    model_path      : Path to .mlpackage produced by dfine_to_coreml.py
    conf_threshold  : Minimum score to keep a detection (per class)
    iou_threshold   : NMS IoU threshold (D-FINE is NMS-free, but kept for compatibility)
    input_size      : Must match what was used during conversion (default 640)
    class_names     : Override with your custom class list if needed
    compute_units   : ct.ComputeUnit.ALL recommended for ANE offload
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.4,
        input_size: int = 640,
        class_names: list = None,
        compute_units=ct.ComputeUnit.ALL,
    ):
        self.conf_threshold = conf_threshold
        self.input_size = input_size
        self.class_names = class_names or COCO_CLASSES
        self.model_path = model_path
        self.compute_units = compute_units

        print(f"Loading CoreML model from {model_path}…")
        self.model = self._load_model(compute_units)
        print("Model loaded. Warming up (3 runs)…")
        dummy = Image.fromarray(
            np.zeros((input_size, input_size, 3), dtype=np.uint8)
        )
        for _ in range(3):
            self._predict_with_fallback(dummy)
        print("Ready.")

    @staticmethod
    def _is_ane_compile_error(exc: Exception) -> bool:
        msg = str(exc)
        return (
            "MILCompilerForANE" in msg
            or "ANECCompile() FAILED" in msg
            or "_ANECompiler" in msg
            or "failed to compile ANE model" in msg
        )

    def _load_model(self, compute_units):
        self.compute_units = compute_units
        return ct.models.MLModel(
            self.model_path,
            compute_units=compute_units,
        )

    def _predict_with_fallback(self, pil_img):
        try:
            return self.model.predict({"image": pil_img})
        except Exception as e:
            if not self._is_ane_compile_error(e):
                raise

            # Fallback to CPU+GPU for models that fail ANE compile at runtime.
            if self.compute_units in (ct.ComputeUnit.ALL, ct.ComputeUnit.CPU_AND_NE):
                print("ANE compile failed at runtime. Falling back to CPU_AND_GPU…")
                self.model = self._load_model(ct.ComputeUnit.CPU_AND_GPU)
                return self.model.predict({"image": pil_img})
            raise

    # ── Core prediction ───────────────────────────────────────────
    def predict(self, image_source, orig_size: tuple = None):
        """
        Parameters
        ----------
        image_source : str path, PIL.Image, or np.ndarray (H,W,3) uint8
        orig_size    : (H, W) of original image for coordinate rescaling.
                       If None, uses input_size.

        Returns
        -------
        list of dicts with keys: box_xyxy, score, class_id, class_name
        """
        pil_img, (oh, ow) = self._load_image(image_source)
        if orig_size:
            oh, ow = orig_size

        t0 = time.perf_counter()
        out = self._predict_with_fallback(pil_img)
        latency_ms = (time.perf_counter() - t0) * 1000

        # out["boxes"]  shape: [1, Q, 4]  (cx, cy, w, h) normalized
        # out["scores"] shape: [1, Q, C]
        boxes  = np.array(out["boxes"])[0]   # [Q, 4]
        scores = np.array(out["scores"])[0]  # [Q, C]

        detections = self._decode(boxes, scores, oh, ow)
        return detections, latency_ms

    def _load_image(self, source):
        if isinstance(source, str):
            img = Image.open(source).convert("RGB")
        elif isinstance(source, np.ndarray):
            img = Image.fromarray(source.astype(np.uint8))
        elif isinstance(source, Image.Image):
            img = source.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(source)}")

        orig_h, orig_w = img.size[1], img.size[0]
        resized = img.resize((self.input_size, self.input_size), Image.BILINEAR)
        return resized, (orig_h, orig_w)

    def _decode(self, boxes, scores, orig_h, orig_w):
        """Convert model outputs → human-friendly detection list."""
        class_ids = np.argmax(scores, axis=-1)       # [Q]
        max_scores = scores[np.arange(len(scores)), class_ids]  # [Q]

        keep = max_scores >= self.conf_threshold
        boxes    = boxes[keep]
        class_ids = class_ids[keep]
        max_scores = max_scores[keep]

        detections = []
        for box, cls_id, score in zip(boxes, class_ids, max_scores):
            cx, cy, w, h = box
            # convert cx/cy/w/h (normalized) → x1/y1/x2/y2 (pixel)
            x1 = (cx - w / 2) * orig_w
            y1 = (cy - h / 2) * orig_h
            x2 = (cx + w / 2) * orig_w
            y2 = (cy + h / 2) * orig_h
            detections.append({
                "box_xyxy": [
                    float(np.clip(x1, 0, orig_w)),
                    float(np.clip(y1, 0, orig_h)),
                    float(np.clip(x2, 0, orig_w)),
                    float(np.clip(y2, 0, orig_h)),
                ],
                "score": float(score),
                "class_id": int(cls_id),
                "class_name": self.class_names[int(cls_id)]
                               if int(cls_id) < len(self.class_names)
                               else str(cls_id),
            })
        return detections

    # ── Visualization ─────────────────────────────────────────────
    def draw(self, detections, image_source, out_path: str = "result.jpg"):
        try:
            import cv2
        except ImportError:
            print("pip install opencv-python for visualization")
            return

        if isinstance(image_source, str):
            img = cv2.imread(image_source)
        else:
            img = np.array(image_source)[:, :, ::-1].copy()

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["box_xyxy"]]
            label = f"{det['class_name']} {det['score']:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(img, label, (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)

        cv2.imwrite(out_path, img)
        print(f"Saved visualization: {out_path}")

    # ── Video / webcam loop ───────────────────────────────────────
    def run_video(self, source=0, display: bool = True):
        """
        source=0 → webcam; source="file.mp4" → video file.
        Shows live FPS powered by CoreML.
        """
        try:
            import cv2
        except ImportError:
            print("pip install opencv-python for video inference")
            return

        cap = cv2.VideoCapture(source)
        fps_buf = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            dets, ms = self.predict(rgb, orig_size=(frame.shape[0], frame.shape[1]))
            fps_buf.append(1000 / ms)
            if len(fps_buf) > 30:
                fps_buf.pop(0)
            avg_fps = sum(fps_buf) / len(fps_buf)

            # Draw
            for det in dets:
                x1, y1, x2, y2 = [int(v) for v in det["box_xyxy"]]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cv2.putText(frame, f"{det['class_name']} {det['score']:.2f}",
                            (x1, max(y1 - 5, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)

            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if display:
                cv2.imshow("D-FINE CoreML", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        if display:
            cv2.destroyAllWindows()
        print(f"Average FPS: {sum(fps_buf)/len(fps_buf):.1f}")