import logging
from typing import List, Optional
from dataclasses import dataclass

import numpy as np

import config

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Represents a single person detection."""
    bbox: tuple  # (x1, y1, x2, y2)
    confidence: float
    class_id: int = 0  # Person class

    @property
    def center(self) -> tuple:
        """Get center point of detection."""
        return (
            (self.bbox[0] + self.bbox[2]) // 2,
            (self.bbox[1] + self.bbox[3]) // 2
        )

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_xyxy(self) -> np.ndarray:
        """Convert to numpy array [x1, y1, x2, y2]."""
        return np.array(self.bbox)

    def to_xywh(self) -> np.ndarray:
        """Convert to [x_center, y_center, width, height]."""
        cx, cy = self.center
        return np.array([cx, cy, self.width, self.height])


class PersonDetector:
    """
    YOLOv8n-based person detector with automatic ONNX optimization.

    On first run, auto-exports the .pt model to ONNX format for 2-3x
    faster CPU inference.  Subsequent runs load the cached ONNX model
    directly.

    ONNX Runtime is pinned to a configurable number of threads to avoid
    saturating the CPU.
    """

    def __init__(
        self,
        model_path: str = None,
        confidence_threshold: float = None,
        device: str = None
    ):
        self.model_path = model_path or config.YOLO_MODEL
        self.confidence_threshold = confidence_threshold or config.PERSON_CONFIDENCE_THRESHOLD
        self.device = device  # None = auto-select (GPU if available)

        self.model = None
        self._initialized = False
        self._use_onnx = False

        # Number of threads for ONNX Runtime inference (limits CPU usage)
        self._onnx_threads = getattr(config, "ONNX_THREADS", 2)

    def initialize(self) -> bool:
        """Load the YOLO model, preferring ONNX if available."""
        try:
            # Try ONNX first for lighter CPU usage
            if self._try_load_onnx():
                return True

            # Fall back to PyTorch YOLO and optionally export ONNX
            return self._load_pytorch()

        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            return False

    def _try_load_onnx(self) -> bool:
        """Try to load or export an ONNX version of the model."""
        import os
        onnx_path = self.model_path.replace(".pt", ".onnx")

        # If ONNX model doesn't exist yet, try to export it
        if not os.path.exists(onnx_path):
            try:
                logger.info(f"Exporting YOLO to ONNX for faster CPU inference...")
                from ultralytics import YOLO
                temp_model = YOLO(self.model_path)
                temp_model.export(
                    format="onnx",
                    imgsz=config.YOLO_IMGSZ,
                    simplify=True,
                    opset=17,
                    half=False,
                )
                del temp_model
                logger.info(f"ONNX model exported: {onnx_path}")
            except Exception as e:
                logger.warning(f"ONNX export failed ({e}), falling back to PyTorch")
                return False

        # Load with ONNX Runtime
        try:
            import onnxruntime as ort
            sess_opts = ort.SessionOptions()
            sess_opts.inter_op_num_threads = 1
            sess_opts.intra_op_num_threads = self._onnx_threads
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self._ort_session = ort.InferenceSession(onnx_path, sess_opts)
            self._ort_input_name = self._ort_session.get_inputs()[0].name
            self._ort_input_shape = self._ort_session.get_inputs()[0].shape  # [1,3,H,W]
            self._use_onnx = True
            self._initialized = True
            logger.info(
                f"ONNX model loaded: {onnx_path} "
                f"(threads={self._onnx_threads})"
            )
            return True
        except ImportError:
            logger.info("onnxruntime not installed, using PyTorch YOLO. "
                        "Install for 2-3x speedup: python -m pip install onnxruntime")
            return False
        except Exception as e:
            logger.warning(f"Failed to load ONNX model ({e}), falling back to PyTorch")
            return False

    def _load_pytorch(self) -> bool:
        """Load the PyTorch YOLO model (fallback)."""
        from ultralytics import YOLO
        logger.info(f"Loading YOLO model: {self.model_path}")
        self.model = YOLO(self.model_path)

        if self.device:
            self.model.to(self.device)

        self._initialized = True
        logger.info("YOLO model loaded successfully (PyTorch)")
        return True

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect persons in a frame.

        Args:
            frame: BGR image as numpy array

        Returns:
            List of Detection objects for persons found
        """
        if not self._initialized:
            logger.warning("Detector not initialized")
            return []

        if self._use_onnx:
            return self._detect_onnx(frame)
        else:
            return self._detect_pytorch(frame)

    def _detect_onnx(self, frame: np.ndarray) -> List[Detection]:
        """Run inference via ONNX Runtime (faster on CPU).

        Uses fully vectorized NumPy post-processing instead of Python
        loops for ~5-10x faster filtering of the 8400-row output grid.
        """
        import cv2
        try:
            # Preprocess: resize, normalize, transpose to NCHW
            h_orig, w_orig = frame.shape[:2]
            imgsz = config.YOLO_IMGSZ
            img = cv2.resize(frame, (imgsz, imgsz))
            img = img.astype(np.float32) / 255.0
            img = img.transpose(2, 0, 1)   # HWC -> CHW
            img = np.expand_dims(img, 0)    # add batch dim

            # Inference
            outputs = self._ort_session.run(None, {self._ort_input_name: img})
            preds = outputs[0]  # shape: [1, N, 5+num_classes] or [1, 5+C, N]

            # YOLOv8 ONNX output is [1, 84, 8400] (transposed)
            if preds.shape[1] < preds.shape[2]:
                preds = preds.transpose(0, 2, 1)  # -> [1, 8400, 84]

            preds = preds[0]  # remove batch dim -> [8400, 84]

            # ── Vectorized post-processing ──
            # Extract person-class confidence and filter
            person_scores = preds[:, 4 + config.PERSON_CLASS_ID]
            mask = person_scores >= self.confidence_threshold
            # Also ensure person class is the argmax class for each row
            best_class = np.argmax(preds[:, 4:], axis=1)
            mask &= (best_class == config.PERSON_CLASS_ID)

            if not np.any(mask):
                return []

            filtered = preds[mask]
            confs = person_scores[mask]

            # Convert cxcywh → x1y1x2y2, scaled to original frame
            cx = filtered[:, 0]
            cy = filtered[:, 1]
            bw = filtered[:, 2]
            bh = filtered[:, 3]
            scale_x = w_orig / imgsz
            scale_y = h_orig / imgsz

            x1 = np.clip(((cx - bw / 2) * scale_x).astype(int), 0, w_orig)
            y1 = np.clip(((cy - bh / 2) * scale_y).astype(int), 0, h_orig)
            x2 = np.clip(((cx + bw / 2) * scale_x).astype(int), 0, w_orig)
            y2 = np.clip(((cy + bh / 2) * scale_y).astype(int), 0, h_orig)

            detections = [
                Detection(bbox=(int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])),
                          confidence=float(confs[i]),
                          class_id=config.PERSON_CLASS_ID)
                for i in range(len(confs))
            ]

            # NMS (simple greedy)
            detections = self._nms(detections, iou_thresh=0.5)
            return detections

        except Exception as e:
            logger.error(f"ONNX detection error: {e}")
            return []

    def _detect_pytorch(self, frame: np.ndarray) -> List[Detection]:
        """Run inference via PyTorch YOLO (fallback)."""
        try:
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                classes=[config.PERSON_CLASS_ID],
                verbose=False,
                imgsz=config.YOLO_IMGSZ,
            )

            detections = []

            for result in results:
                if result.boxes is None:
                    continue

                boxes = result.boxes
                for i in range(len(boxes)):
                    box = boxes[i]

                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())

                    if cls == config.PERSON_CLASS_ID:
                        detection = Detection(
                            bbox=(int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])),
                            confidence=conf,
                            class_id=cls
                        )
                        detections.append(detection)

            return detections

        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []

    @staticmethod
    def _nms(detections: List[Detection], iou_thresh: float = 0.5) -> List[Detection]:
        """Simple greedy NMS."""
        if not detections:
            return []

        detections.sort(key=lambda d: d.confidence, reverse=True)
        kept = []

        for det in detections:
            overlap = False
            for k in kept:
                iou = PersonDetector._iou(det.bbox, k.bbox)
                if iou > iou_thresh:
                    overlap = True
                    break
            if not overlap:
                kept.append(det)

        return kept

    @staticmethod
    def _iou(a, b) -> float:
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def detect_with_scores(self, frame: np.ndarray) -> tuple:
        """
        Detect persons and return in format suitable for tracker.

        Returns:
            Tuple of (detections_array, scores_array) where:
            - detections_array: np.ndarray of shape (N, 4) with [x1, y1, x2, y2]
            - scores_array: np.ndarray of shape (N,) with confidence scores
        """
        detections = self.detect(frame)

        if not detections:
            return np.empty((0, 4)), np.empty(0)

        boxes = np.array([d.bbox for d in detections])
        scores = np.array([d.confidence for d in detections])

        return boxes, scores

    @property
    def is_initialized(self) -> bool:
        return self._initialized
