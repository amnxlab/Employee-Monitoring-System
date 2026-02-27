import cv2
import logging
from typing import Optional, Tuple

import config

logger = logging.getLogger(__name__)


class Camera:
    """
    Simple camera interface for OBSBOT Tiny 2 (or any UVC webcam).

    All PTZ/movement control has been removed. The camera is treated as a fixed
    sensor that continuously provides frames for detection and tracking.
    """

    def __init__(
        self,
        camera_index: int = None,
        width: int = None,
        height: int = None,
    ):
        self.camera_index = camera_index if camera_index is not None else config.CAMERA_INDEX
        self.width = width if width is not None else config.FRAME_WIDTH
        self.height = height if height is not None else config.FRAME_HEIGHT

        self.cap: Optional[cv2.VideoCapture] = None
        self._frame_center: Tuple[int, int] = (self.width // 2, self.height // 2)

    def initialize(self) -> bool:
        """Initialize camera."""
        try:
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)

            if not self.cap.isOpened():
                logger.error(f"Failed to open camera at index {self.camera_index}")
                return False

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

            target_fps = getattr(config, "CAMERA_FPS", 30)
            self.cap.set(cv2.CAP_PROP_FPS, target_fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            self._frame_center = (actual_width // 2, actual_height // 2)

            logger.info(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps:.0f}fps")
            logger.info("Camera movement/PTZ control is disabled; using static view only.")

            return True

        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            return False

    def read(self) -> Optional[cv2.typing.MatLike]:
        """Read a frame from the camera."""
        if self.cap is None or not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        """Release camera resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        logger.info("Camera released")

    @property
    def frame_center(self) -> Tuple[int, int]:
        """Get the center point of the frame."""
        return self._frame_center

    @property
    def is_open(self) -> bool:
        """Check if camera is open and ready."""
        return self.cap is not None and self.cap.isOpened()
