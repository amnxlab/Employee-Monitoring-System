"""
Multi-camera manager – spawns one threaded frame reader per camera.

Usage:
    mgr = CameraManager(camera_indices=[0, 1])
    mgr.initialize()
    frames = mgr.get_latest_frames()   # {0: np.ndarray, 1: np.ndarray}
    mgr.release()
"""

import sys
import threading
import time
import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import config
from .camera import Camera

logger = logging.getLogger(__name__)


class _CameraThread:
    """Background thread that continuously reads frames from a single camera."""

    def __init__(self, cam_id: int, camera: Camera):
        self.cam_id = cam_id
        self.camera = camera
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        # Throttle reads to the camera's FPS to avoid busy-spinning
        cam_fps = getattr(config, "CAMERA_FPS", 30)
        self._frame_interval = 1.0 / max(cam_fps, 1)

    # ── lifecycle ──────────────────────────────────────────────────────

    def start(self):
        self._running = True
        self._thread = threading.Thread(
            target=self._read_loop, daemon=True, name=f"cam-{self.cam_id}"
        )
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    # ── public ─────────────────────────────────────────────────────────

    @property
    def latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._latest_frame

    # ── internal ───────────────────────────────────────────────────────

    def _read_loop(self):
        while self._running:
            t0 = time.perf_counter()
            frame = self.camera.read()
            if frame is not None:
                with self._lock:
                    self._latest_frame = frame
            else:
                time.sleep(0.03)  # back off on read failure
                continue
            # Sleep for the remainder of the frame interval
            elapsed = time.perf_counter() - t0
            sleep_time = self._frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


class CameraManager:
    """
    Manages multiple cameras, each running on its own thread.

    Provides synchronised access to the latest frame from every camera.
    """

    def __init__(self, camera_indices: Optional[List[int]] = None):
        self.camera_indices = camera_indices or getattr(config, "CAMERA_INDICES", [0])
        self._threads: Dict[int, _CameraThread] = {}
        self._cameras: Dict[int, Camera] = {}

    # ── lifecycle ──────────────────────────────────────────────────────

    def initialize(self) -> bool:
        """Open every camera and start reader threads."""
        success = True
        for idx in self.camera_indices:
            cam = Camera(camera_index=idx)
            if not cam.initialize():
                logger.error(f"Failed to initialize camera {idx}")
                success = False
                continue
            self._cameras[idx] = cam
            ct = _CameraThread(cam_id=idx, camera=cam)
            ct.start()
            self._threads[idx] = ct
            logger.info(f"Camera {idx} thread started")

        if not self._cameras:
            logger.error("No cameras initialised")
            return False

        logger.info(
            f"CameraManager ready – {len(self._cameras)} camera(s): "
            f"{list(self._cameras.keys())}"
        )
        return success

    def release(self):
        """Stop all threads and release all cameras."""
        for ct in self._threads.values():
            ct.stop()
        for cam in self._cameras.values():
            cam.release()
        self._threads.clear()
        self._cameras.clear()
        logger.info("CameraManager released all cameras")

    # ── frame access ───────────────────────────────────────────────────

    def get_latest_frames(self) -> Dict[int, np.ndarray]:
        """Return {cam_id: frame} for every camera that has a frame ready."""
        frames: Dict[int, np.ndarray] = {}
        for cam_id, ct in self._threads.items():
            f = ct.latest_frame
            if f is not None:
                frames[cam_id] = f
        return frames

    def get_camera_ids(self) -> List[int]:
        return list(self._cameras.keys())

    @property
    def num_cameras(self) -> int:
        return len(self._cameras)
