"""
Global cross-camera ID binder with persistent body tracking.

Maintains a unified employee -> (cam_id, track_id, descriptor) map across
all cameras.  Supports:
  - face-recognition-based binding (initial identification)
  - cross-camera histogram handoff (no face recognition needed)
  - same-camera re-identification via body descriptor
  - spatial memory (position-based matching boost)
  - 2-hour binding persistence (auto-bind without face rec)

After initial face recognition, employees are tracked purely by body
appearance + position for BINDING_PERSIST_SECONDS (default 2 hours).
"""

import math
import time
import logging
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import cv2
import numpy as np

import config
from .tracker import Track
from .face_recognition import FaceDetection
from .id_binder import IDBinder, Binding

logger = logging.getLogger(__name__)

# ── body descriptor helpers ────────────────────────────────────────────


def _body_crop(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """Extract body crop from frame, skipping the top 20% (head)."""
    x1, y1, x2, y2 = bbox
    h = y2 - y1
    crop_y1 = y1 + int(h * 0.2)  # skip head region
    crop = frame[max(crop_y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)]
    return crop


def _compute_histogram(crop: np.ndarray) -> Optional[np.ndarray]:
    """Compute a normalised HSV colour histogram for a body crop."""
    if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def _compare_histograms(h1: np.ndarray, h2: np.ndarray) -> float:
    """Return correlation score in [-1, 1]; higher = more similar."""
    return float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))


# ── multi-feature body descriptor ─────────────────────────────────────


@dataclass
class BodyDescriptor:
    """Lightweight appearance descriptor for a tracked body (~0.2ms to compute).

    Combines multiple cheap features for robust matching without a neural net.
    """
    histogram: Optional[np.ndarray] = None   # HSV colour histogram
    aspect_ratio: float = 0.0                # width / height of body bbox
    avg_upper_color: Optional[np.ndarray] = None  # mean BGR of upper body half
    avg_lower_color: Optional[np.ndarray] = None  # mean BGR of lower body half
    position: Tuple[int, int] = (0, 0)       # last known bbox center
    timestamp: float = 0.0                    # when this descriptor was updated


def compute_body_descriptor(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> BodyDescriptor:
    """Build a BodyDescriptor from a frame crop.  ~0.2ms on CPU."""
    x1, y1, x2, y2 = bbox
    w = max(x2 - x1, 1)
    h = max(y2 - y1, 1)
    center = ((x1 + x2) // 2, (y1 + y2) // 2)

    crop = _body_crop(frame, bbox)
    hist = _compute_histogram(crop)

    # Split body into upper/lower halves for colour feature
    avg_upper = None
    avg_lower = None
    if crop.size > 0 and crop.shape[0] >= 10:
        mid = crop.shape[0] // 2
        upper_half = crop[:mid]
        lower_half = crop[mid:]
        if upper_half.size > 0:
            avg_upper = upper_half.mean(axis=(0, 1)).astype(np.float32)
        if lower_half.size > 0:
            avg_lower = lower_half.mean(axis=(0, 1)).astype(np.float32)

    return BodyDescriptor(
        histogram=hist,
        aspect_ratio=w / h,
        avg_upper_color=avg_upper,
        avg_lower_color=avg_lower,
        position=center,
        timestamp=time.time(),
    )


def compare_descriptors(a: BodyDescriptor, b: BodyDescriptor, is_cross_camera: bool = False) -> float:
    """
    Compare two body descriptors.  Returns a score in [0, 1].

    When is_cross_camera=True the spatial-proximity component is
    disabled (pixels from different camera views are unrelated) and
    its weight is redistributed to the histogram component.

    Same-camera scoring breakdown (weights sum to 1.0):
      0.45  histogram correlation
      0.15  aspect ratio similarity
      0.15  upper body colour distance
      0.15  lower body colour distance
      0.10  spatial proximity

    Cross-camera scoring breakdown:
      0.55  histogram correlation  (+0.10 from spatial)
      0.15  aspect ratio similarity
      0.15  upper body colour distance
      0.15  lower body colour distance
      0.00  spatial proximity (disabled)
    """
    score = 0.0

    hist_weight = 0.55 if is_cross_camera else 0.45
    spatial_weight = 0.0 if is_cross_camera else 0.10

    # Histogram
    if a.histogram is not None and b.histogram is not None:
        hist_corr = _compare_histograms(a.histogram, b.histogram)
        score += hist_weight * max(0.0, hist_corr)

    # Aspect ratio similarity (0.15) -- penalise large differences
    ar_diff = abs(a.aspect_ratio - b.aspect_ratio)
    ar_score = max(0.0, 1.0 - ar_diff * 3.0)  # 0.33 diff -> 0 score
    score += 0.15 * ar_score

    # Upper body colour (0.15)
    if a.avg_upper_color is not None and b.avg_upper_color is not None:
        upper_dist = float(np.linalg.norm(a.avg_upper_color - b.avg_upper_color))
        upper_score = max(0.0, 1.0 - upper_dist / 100.0)
        score += 0.15 * upper_score

    # Lower body colour (0.15)
    if a.avg_lower_color is not None and b.avg_lower_color is not None:
        lower_dist = float(np.linalg.norm(a.avg_lower_color - b.avg_lower_color))
        lower_score = max(0.0, 1.0 - lower_dist / 100.0)
        score += 0.15 * lower_score

    # Spatial proximity (same-camera only)
    if spatial_weight > 0:
        spatial_px = getattr(config, "SPATIAL_MATCH_PIXELS", 150)
        px_dist = math.hypot(a.position[0] - b.position[0], a.position[1] - b.position[1])
        spatial_score = max(0.0, 1.0 - px_dist / spatial_px)
        score += spatial_weight * spatial_score

    return score


# ── data classes ───────────────────────────────────────────────────────


@dataclass
class EmployeeBinding:
    """Tracks where an employee is currently seen (or was last seen)."""
    employee_id: str
    cam_id: int
    track_id: int
    descriptor: Optional[BodyDescriptor] = None
    last_seen: float = 0.0
    active: bool = True

    # Legacy compat: expose histogram directly
    @property
    def histogram(self) -> Optional[np.ndarray]:
        return self.descriptor.histogram if self.descriptor else None


# ── main class ─────────────────────────────────────────────────────────


class GlobalIDBinder:
    """
    Cross-camera identity manager with persistent body tracking.

    - One *per-camera* ``IDBinder`` handles local face<->track binding.
    - This class maintains the global mapping, cross/same-camera handoff,
      and long-term body descriptor memory.

    After initial face recognition, employees are tracked for up to
    BINDING_PERSIST_SECONDS purely by body appearance + position.
    """

    def __init__(self):
        self._per_cam_binders: Dict[int, IDBinder] = {}

        # Thread lock for shared data structures (main thread + face-rec thread)
        self._lock = threading.Lock()

        # employee_id -> EmployeeBinding (currently active)
        self._employee_bindings: Dict[str, EmployeeBinding] = {}

        # (cam_id, track_id) -> employee_id
        self._track_to_employee: Dict[Tuple[int, int], str] = {}

        # Remembered employees for persistent re-ID (face rec not needed)
        # employee_id -> EmployeeBinding snapshot at time of loss
        self._remembered: Dict[str, EmployeeBinding] = {}

        self._handoff_window = getattr(config, "HANDOFF_WINDOW_SECONDS", 30)
        self._hist_threshold = getattr(config, "HISTOGRAM_MATCH_THRESHOLD", 0.7)
        self._persist_seconds = getattr(config, "BINDING_PERSIST_SECONDS", 7200)
        self._same_cam_window = getattr(config, "SAME_CAM_HANDOFF_WINDOW", 300)

        # Descriptor match thresholds (0-1):
        # Same-camera re-ID can use a lower threshold (colour is more stable)
        self._descriptor_threshold_same = 0.60
        # Cross-camera needs a higher bar (different lighting, angle)
        self._descriptor_threshold_cross = 0.65
        # Minimum gap between best and second-best to accept a match
        self._descriptor_gap_min = 0.08

    # ── per-camera binder management ───────────────────────────────────

    def register_camera(self, cam_id: int):
        """Create a local IDBinder for a camera."""
        if cam_id not in self._per_cam_binders:
            self._per_cam_binders[cam_id] = IDBinder()
            logger.info(f"Registered local binder for camera {cam_id}")

    def get_local_binder(self, cam_id: int) -> IDBinder:
        return self._per_cam_binders[cam_id]

    # ── binding (from face recognition) ────────────────────────────────

    def bind(
        self,
        cam_id: int,
        track_id: int,
        employee_id: str,
        frame: Optional[np.ndarray] = None,
        bbox: Optional[Tuple[int, int, int, int]] = None,
    ):
        """Bind a track on a specific camera to an employee.

        Refuses to overwrite an existing (cam, track)->employee mapping
        with a *different* employee (defensive guard against identity swaps).
        """
        now = time.time()

        # Compute body descriptor outside the lock (CPU-intensive)
        descriptor = None
        if frame is not None and bbox is not None:
            descriptor = compute_body_descriptor(frame, bbox)

        with self._lock:
            # Defensive guard: if this track is already bound to a DIFFERENT
            # employee, refuse the re-bind to prevent identity swaps.
            existing_emp = self._track_to_employee.get((cam_id, track_id))
            if existing_emp is not None and existing_emp != employee_id:
                logger.warning(
                    f"Refusing re-bind: cam {cam_id} track {track_id} is "
                    f"already bound to {existing_emp}, rejecting {employee_id}"
                )
                return

            # Remove any previous binding for this employee on another camera
            old = self._employee_bindings.get(employee_id)
            if old and (old.cam_id != cam_id or old.track_id != track_id):
                self._track_to_employee.pop((old.cam_id, old.track_id), None)

            binding = EmployeeBinding(
                employee_id=employee_id,
                cam_id=cam_id,
                track_id=track_id,
                descriptor=descriptor,
                last_seen=now,
                active=True,
            )
            self._employee_bindings[employee_id] = binding
            self._track_to_employee[(cam_id, track_id)] = employee_id

            # Also update the local per-camera binder
            local = self._per_cam_binders.get(cam_id)
            if local:
                local.bind(track_id, employee_id)

            # Remove from remembered if re-appeared
            self._remembered.pop(employee_id, None)

        logger.debug(
            f"Global bind: {employee_id} -> cam {cam_id} track {track_id}"
        )

    # ── persistent re-ID (same-camera + cross-camera) ──────────────────

    def attempt_handoff(
        self,
        cam_id: int,
        track: Track,
        frame: np.ndarray,
    ) -> Optional[str]:
        """
        Try to match an unbound track to a remembered employee using body
        descriptor matching (histogram + colour + aspect ratio + position).

        Works for BOTH cross-camera and same-camera re-identification.

        Returns the matched employee_id or None.
        """
        now = time.time()

        new_desc = compute_body_descriptor(frame, track.bbox)
        if new_desc.histogram is None:
            return None

        best_emp: Optional[str] = None
        best_score = -1.0
        second_best_score = -1.0

        for emp_id, mem_binding in list(self._remembered.items()):
            # Determine max window: same camera gets longer window
            is_same_cam = (mem_binding.cam_id == cam_id)
            if is_same_cam:
                max_window = self._same_cam_window
            else:
                # Cross-camera body handoff disabled when HANDOFF_WINDOW_SECONDS == 0.
                # Cross-camera identity must be established via face recognition only.
                if self._handoff_window == 0:
                    continue
                max_window = self._handoff_window

            # Never exceed the global persistence timeout
            max_window = min(max_window, self._persist_seconds)

            # Check time window
            age = now - mem_binding.last_seen
            if age > max_window:
                # If within persist window, keep in memory but skip matching
                if age > self._persist_seconds:
                    self._remembered.pop(emp_id, None)
                continue

            if mem_binding.descriptor is None:
                continue

            score = compare_descriptors(
                new_desc, mem_binding.descriptor,
                is_cross_camera=(not is_same_cam),
            )

            # Boost score for same-camera matches (more reliable)
            if is_same_cam:
                score = min(1.0, score * 1.1)

            # Pick the appropriate threshold
            threshold = self._descriptor_threshold_same if is_same_cam else self._descriptor_threshold_cross

            if score > threshold and score > best_score:
                second_best_score = best_score
                best_score = score
                best_emp = emp_id
            elif score > second_best_score:
                second_best_score = score

        # Gap check: best must be clearly better than runner-up to avoid
        # ambiguous matches (e.g. two people in similar dark clothing).
        if best_emp is not None:
            gap = best_score - max(0.0, second_best_score)
            if gap < self._descriptor_gap_min:
                logger.debug(
                    f"[AUTO-BIND REJECTED] best={best_score:.3f} "
                    f"second={second_best_score:.3f} gap={gap:.3f} < "
                    f"{self._descriptor_gap_min} — ambiguous, skipping"
                )
                return None

        if best_emp is not None:
            logger.info(
                f"[AUTO-BIND] {best_emp} -> cam {cam_id} track "
                f"{track.track_id} (score={best_score:.3f}, "
                f"{'same' if self._remembered[best_emp].cam_id == cam_id else 'cross'}-cam)"
            )
            self.bind(cam_id, track.track_id, best_emp, frame, track.bbox)

        return best_emp

    # ── lost-track management ──────────────────────────────────────────

    def mark_lost(self, cam_id: int, track_id: int):
        """Called when a bound track is lost on a specific camera."""
        key = (cam_id, track_id)
        with self._lock:
            emp_id = self._track_to_employee.pop(key, None)
            if emp_id is None:
                return

            binding = self._employee_bindings.get(emp_id)
            if binding and binding.cam_id == cam_id and binding.track_id == track_id:
                binding.active = False
                # Save to persistent memory for re-ID (up to BINDING_PERSIST_SECONDS)
                self._remembered[emp_id] = binding
                logger.debug(
                    f"Employee {emp_id} lost on cam {cam_id}, "
                    f"remembered for {self._persist_seconds}s"
                )

    def cleanup_lost_tracks(self, cam_id: int, active_track_ids: List[int]):
        """Remove bindings for tracks no longer active on a camera."""
        with self._lock:
            lost_keys = [
                (c, t)
                for (c, t) in list(self._track_to_employee.keys())
                if c == cam_id and t not in active_track_ids
            ]
        for key in lost_keys:
            self.mark_lost(*key)

        # Also clean local binder
        local = self._per_cam_binders.get(cam_id)
        if local:
            local.cleanup_lost_tracks(active_track_ids)

        # Evict expired remembered bindings to prevent memory leak
        now = time.time()
        expired = [k for k, v in self._remembered.items()
                   if now - v.last_seen > self._persist_seconds]
        for k in expired:
            del self._remembered[k]

    # ── queries ────────────────────────────────────────────────────────

    def get_employee_for_track(self, cam_id: int, track_id: int) -> Optional[str]:
        return self._track_to_employee.get((cam_id, track_id))

    def is_employee_visible_any_camera(self, employee_id: str) -> bool:
        """True if the employee has an active bound track on any camera."""
        binding = self._employee_bindings.get(employee_id)
        if binding is None or not binding.active:
            return False
        # Double-check the track is actually in the track map
        return (binding.cam_id, binding.track_id) in self._track_to_employee

    def get_visible_employees(self) -> List[str]:
        return [
            emp_id
            for emp_id, b in self._employee_bindings.items()
            if b.active and (b.cam_id, b.track_id) in self._track_to_employee
        ]

    def get_unbound_tracks(self, cam_id: int, tracks: List[Track]) -> List[Track]:
        """Tracks on *cam_id* not bound to any employee."""
        return [
            t for t in tracks
            if (cam_id, t.track_id) not in self._track_to_employee
        ]

    def get_all_bindings_for_cam(self, cam_id: int) -> Dict[int, str]:
        """Return {track_id: employee_id} for a specific camera."""
        return {
            t: e
            for (c, t), e in self._track_to_employee.items()
            if c == cam_id
        }

    def has_remembered(self, employee_id: str) -> bool:
        """True if the employee is in the persistent memory pool."""
        return employee_id in self._remembered

    # ── update descriptor for active tracks ────────────────────────────

    def update_histogram(
        self,
        cam_id: int,
        track_id: int,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ):
        """Refresh the stored descriptor for an active binding (called periodically)."""
        # Compute descriptor outside lock (CPU-intensive)
        desc = compute_body_descriptor(frame, bbox)
        if desc.histogram is None:
            return

        with self._lock:
            emp_id = self._track_to_employee.get((cam_id, track_id))
            if emp_id is None:
                return
            binding = self._employee_bindings.get(emp_id)
            if binding is None:
                return
            binding.descriptor = desc
            binding.last_seen = time.time()

    # ── reset ──────────────────────────────────────────────────────────

    def reset(self):
        self._employee_bindings.clear()
        self._track_to_employee.clear()
        self._remembered.clear()
        for binder in self._per_cam_binders.values():
            binder.reset()
        logger.info("Global ID binder reset")
