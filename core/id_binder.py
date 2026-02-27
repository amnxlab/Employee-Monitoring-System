import logging
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import config
from .tracker import Track
from .face_recognition import FaceDetection
from utils.helpers import calculate_iou

logger = logging.getLogger(__name__)


@dataclass
class Binding:
    """Represents a binding between track ID and employee ID."""
    track_id: int
    employee_id: str
    created_at: float
    last_validated: float
    validation_count: int = 1
    
    def validate(self):
        """Update validation timestamp and count."""
        self.last_validated = time.time()
        self.validation_count += 1


@dataclass
class BinderState:
    """Internal state for ID binding."""
    track_to_employee: Dict[int, str] = field(default_factory=dict)
    employee_to_track: Dict[str, int] = field(default_factory=dict)
    bindings: Dict[int, Binding] = field(default_factory=dict)


class IDBinder:
    """
    Manages binding between ByteTrack track IDs and employee IDs.
    
    When a face is recognized, the corresponding person track is bound
    to the employee ID. This binding persists even when the face is
    no longer visible, allowing continued tracking via body detection.
    
    Includes validation to prevent binding non-person objects (hands, etc).
    """
    
    def __init__(self, iou_threshold: float = None):
        self.iou_threshold = iou_threshold or getattr(config, "BINDING_IOU_THRESHOLD", 0.4)
        self.min_track_area = getattr(config, "MIN_TRACK_AREA", 5000)
        self.aspect_ratio_min = getattr(config, "TRACK_ASPECT_RATIO_MIN", 0.25)
        self.aspect_ratio_max = getattr(config, "TRACK_ASPECT_RATIO_MAX", 0.85)
        self.face_upper_ratio = getattr(config, "FACE_IN_UPPER_RATIO", 0.6)
        self.state = BinderState()
    
    def bind(self, track_id: int, employee_id: str):
        """
        Create or update a binding between track ID and employee ID.
        
        Args:
            track_id: ByteTrack track identifier
            employee_id: Employee identifier from face recognition
        """
        current_time = time.time()
        
        old_track = self.state.employee_to_track.get(employee_id)
        if old_track is not None and old_track != track_id:
            self._remove_binding(old_track)
        
        old_employee = self.state.track_to_employee.get(track_id)
        if old_employee is not None and old_employee != employee_id:
            logger.warning(
                f"Track {track_id} reassigned from {old_employee} to {employee_id}"
            )
        
        self.state.track_to_employee[track_id] = employee_id
        self.state.employee_to_track[employee_id] = track_id
        
        if track_id in self.state.bindings:
            self.state.bindings[track_id].validate()
        else:
            self.state.bindings[track_id] = Binding(
                track_id=track_id,
                employee_id=employee_id,
                created_at=current_time,
                last_validated=current_time
            )
        
        logger.debug(f"Bound track {track_id} to employee {employee_id}")
    
    def _remove_binding(self, track_id: int):
        """Remove a binding by track ID."""
        if track_id in self.state.track_to_employee:
            employee_id = self.state.track_to_employee[track_id]
            del self.state.track_to_employee[track_id]
            
            if self.state.employee_to_track.get(employee_id) == track_id:
                del self.state.employee_to_track[employee_id]
            
            if track_id in self.state.bindings:
                del self.state.bindings[track_id]
            
            logger.debug(f"Removed binding for track {track_id}")
    
    def unbind_track(self, track_id: int):
        """Explicitly unbind a track."""
        self._remove_binding(track_id)
    
    def unbind_employee(self, employee_id: str):
        """Explicitly unbind an employee."""
        track_id = self.state.employee_to_track.get(employee_id)
        if track_id is not None:
            self._remove_binding(track_id)
    
    def get_employee_for_track(self, track_id: int) -> Optional[str]:
        """Get employee ID for a track."""
        return self.state.track_to_employee.get(track_id)
    
    def get_track_for_employee(self, employee_id: str) -> Optional[int]:
        """Get track ID for an employee."""
        return self.state.employee_to_track.get(employee_id)
    
    def is_employee_visible(self, employee_id: str, active_track_ids: List[int]) -> bool:
        """
        Check if an employee is currently visible (has an active track).
        
        Args:
            employee_id: Employee to check
            active_track_ids: List of currently active track IDs
            
        Returns:
            True if employee has an active track binding
        """
        track_id = self.state.employee_to_track.get(employee_id)
        if track_id is None:
            return False
        return track_id in active_track_ids
    
    def get_visible_employees(self, active_track_ids: List[int]) -> List[str]:
        """Get list of employees that are currently visible."""
        visible = []
        for employee_id, track_id in self.state.employee_to_track.items():
            if track_id in active_track_ids:
                visible.append(employee_id)
        return visible
    
    def _is_valid_person_track(self, track: Track) -> bool:
        """
        Validate that a track looks like a person (not a hand or random object).
        
        Checks:
        - Minimum area
        - Aspect ratio (persons are taller than wide)
        """
        x1, y1, x2, y2 = track.bbox
        width = x2 - x1
        height = y2 - y1
        area = width * height
        
        if area < self.min_track_area:
            logger.debug(f"Track {track.track_id} rejected: area {area} < {self.min_track_area}")
            return False
        
        if height <= 0:
            return False
        aspect_ratio = width / height
        if aspect_ratio < self.aspect_ratio_min or aspect_ratio > self.aspect_ratio_max:
            logger.debug(f"Track {track.track_id} rejected: aspect ratio {aspect_ratio:.2f} outside [{self.aspect_ratio_min}, {self.aspect_ratio_max}]")
            return False
        
        return True
    
    def _is_face_in_upper_body(self, face_bbox: Tuple[int, int, int, int], track_bbox: Tuple[int, int, int, int]) -> bool:
        """
        Check if face is in the upper portion of the track (head position).
        
        This prevents binding a face to a track where the face is at the bottom
        (which would indicate the track is not the person's body).
        """
        fx1, fy1, fx2, fy2 = face_bbox
        tx1, ty1, tx2, ty2 = track_bbox
        
        face_center_y = (fy1 + fy2) / 2
        track_height = ty2 - ty1
        upper_boundary = ty1 + track_height * self.face_upper_ratio
        
        if face_center_y > upper_boundary:
            logger.debug(f"Face rejected: center Y {face_center_y:.0f} below upper boundary {upper_boundary:.0f}")
            return False
        
        return True
    
    def _is_face_inside_track(self, face_bbox: Tuple[int, int, int, int], track_bbox: Tuple[int, int, int, int]) -> bool:
        """Check if face center is inside the track bbox (simple containment check)."""
        fx1, fy1, fx2, fy2 = face_bbox
        tx1, ty1, tx2, ty2 = track_bbox
        
        face_cx = (fx1 + fx2) / 2
        face_cy = (fy1 + fy2) / 2
        
        return tx1 <= face_cx <= tx2 and ty1 <= face_cy <= ty2

    def find_overlapping_track(
        self,
        face_bbox: Tuple[int, int, int, int],
        tracks: List[Track]
    ) -> Optional[int]:
        """
        Find the track that contains the detected face.
        
        Uses simple containment check - face center must be inside track bbox.
        """
        best_track_id = None
        best_area = 0
        
        for track in tracks:
            # Check if face is inside this track
            if not self._is_face_inside_track(face_bbox, track.bbox):
                logger.debug(f"Track {track.track_id}: face not inside track bbox")
                continue
            
            # Prefer larger tracks (more likely to be full body)
            area = track.area
            if area > best_area:
                best_area = area
                best_track_id = track.track_id
        
        return best_track_id
    
    def process_face_matches(
        self,
        faces: List[FaceDetection],
        face_to_employee: Dict[int, str],
        tracks: List[Track]
    ):
        """
        Process face recognition results and create bindings.
        
        Args:
            faces: List of detected faces
            face_to_employee: Mapping of face index to employee ID
            tracks: Current tracks from tracker
        """
        for face_idx, employee_id in face_to_employee.items():
            face = faces[face_idx]
            track_id = self.find_overlapping_track(face.bbox, tracks)
            
            if track_id is not None:
                self.bind(track_id, employee_id)
                logger.info(f"Bound track {track_id} to employee {employee_id}")
            else:
                logger.warning(f"No valid track found for face of {employee_id} (tracks: {len(tracks)})")
    
    def cleanup_lost_tracks(self, active_track_ids: List[int]):
        """Remove bindings for tracks that are no longer active."""
        lost_tracks = [
            track_id for track_id in self.state.track_to_employee.keys()
            if track_id not in active_track_ids
        ]
        
        for track_id in lost_tracks:
            self._remove_binding(track_id)
    
    def get_binding_info(self, track_id: int) -> Optional[Binding]:
        """Get binding information for a track."""
        return self.state.bindings.get(track_id)
    
    def get_all_bindings(self) -> Dict[int, str]:
        """Get all current track-to-employee bindings."""
        return dict(self.state.track_to_employee)
    
    def get_unbound_tracks(self, tracks: List[Track]) -> List[Track]:
        """Get tracks that are not yet bound to any employee."""
        return [t for t in tracks if t.track_id not in self.state.track_to_employee]
    
    def get_bound_track_ids(self) -> List[int]:
        """Get list of track IDs that are currently bound."""
        return list(self.state.track_to_employee.keys())
    
    def reset(self):
        """Reset all bindings."""
        self.state = BinderState()
        logger.info("ID binder reset")
