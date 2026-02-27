"""
PTZ Controller - disabled

All camera movement / PTZ logic has been removed from this project.
This stub exists only so existing imports continue to work.
"""
from typing import List, Optional, Tuple

from core.camera import Camera
from core.tracker import Track
from core.id_binder import IDBinder


class PTZController:
    """No-op PTZ controller stub."""

    def __init__(self, camera: Camera):
        self.camera = camera

    def update(
        self,
        tracks: List[Track],
        id_binder: IDBinder,
        frame_shape: Tuple[int, int],
    ):
        return

    def get_current_target(self) -> Optional[int]:
        return None

    def get_identified_count(self) -> int:
        return 0

    def is_sweep_routine_active(self) -> bool:
        return False

    def get_sweep_routine_status(self) -> Optional[Tuple[int, int, float]]:
        return None

    def get_next_sweep_time(self) -> float:
        return 0.0

    def force_sweep_routine(self):
        return

    def force_scan(self):
        return

    def go_home(self):
        return

    def set_enabled(self, enabled: bool):
        return
