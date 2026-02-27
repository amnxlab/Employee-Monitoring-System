import logging
from typing import List, Optional, Dict
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import supervision as sv

import config

logger = logging.getLogger(__name__)


class TrackStatus(Enum):
    NEW = "new"
    TRACKED = "tracked"
    LOST = "lost"


@dataclass
class Track:
    """Represents a tracked person."""
    track_id: int
    bbox: tuple  # (x1, y1, x2, y2)
    confidence: float
    status: TrackStatus = TrackStatus.NEW
    frames_tracked: int = 1
    frames_lost: int = 0
    
    @property
    def center(self) -> tuple:
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


@dataclass
class TrackerState:
    """Internal state for tracking."""
    active_tracks: Dict[int, Track] = field(default_factory=dict)
    lost_tracks: Dict[int, int] = field(default_factory=dict)  # track_id -> frames_lost


class PersonTracker:
    """
    ByteTrack-based multi-object tracker for persons.
    Uses supervision library for ByteTrack implementation.
    """
    
    def __init__(self, track_buffer: int = None):
        self.track_buffer = track_buffer or config.TRACK_BUFFER
        
        self.tracker: Optional[sv.ByteTrack] = None
        self.state = TrackerState()
        self._initialized = False
        self._previous_track_ids: set = set()
    
    def initialize(self) -> bool:
        """Initialize the ByteTrack tracker."""
        try:
            self.tracker = sv.ByteTrack(
                track_activation_threshold=0.25,
                lost_track_buffer=self.track_buffer,
                minimum_matching_threshold=0.8,
                frame_rate=30
            )
            
            self._initialized = True
            logger.info("ByteTrack tracker initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize tracker: {e}")
            return False
    
    def update(self, detections: np.ndarray, scores: np.ndarray) -> List[Track]:
        """
        Update tracker with new detections.
        
        Args:
            detections: np.ndarray of shape (N, 4) with [x1, y1, x2, y2]
            scores: np.ndarray of shape (N,) with confidence scores
            
        Returns:
            List of Track objects with current tracks
        """
        if not self._initialized or self.tracker is None:
            return []
        
        try:
            if len(detections) == 0:
                sv_detections = sv.Detections.empty()
            else:
                sv_detections = sv.Detections(
                    xyxy=detections,
                    confidence=scores
                )
            
            tracked = self.tracker.update_with_detections(sv_detections)
            
            current_track_ids = set()
            tracks = []
            
            if tracked.tracker_id is not None:
                for i, track_id in enumerate(tracked.tracker_id):
                    track_id = int(track_id)
                    current_track_ids.add(track_id)
                    
                    bbox = tracked.xyxy[i]
                    conf = tracked.confidence[i] if tracked.confidence is not None else 1.0
                    
                    if track_id in self.state.active_tracks:
                        status = TrackStatus.TRACKED
                        frames = self.state.active_tracks[track_id].frames_tracked + 1
                    elif track_id in self._previous_track_ids:
                        status = TrackStatus.TRACKED
                        frames = 1
                    else:
                        status = TrackStatus.NEW
                        frames = 1
                    
                    track = Track(
                        track_id=track_id,
                        bbox=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                        confidence=float(conf),
                        status=status,
                        frames_tracked=frames
                    )
                    tracks.append(track)
                    self.state.active_tracks[track_id] = track
            
            lost_ids = self._previous_track_ids - current_track_ids
            for lost_id in lost_ids:
                if lost_id in self.state.lost_tracks:
                    self.state.lost_tracks[lost_id] += 1
                else:
                    self.state.lost_tracks[lost_id] = 1
                
                if self.state.lost_tracks[lost_id] > self.track_buffer:
                    if lost_id in self.state.active_tracks:
                        del self.state.active_tracks[lost_id]
                    del self.state.lost_tracks[lost_id]
            
            for track_id in current_track_ids:
                if track_id in self.state.lost_tracks:
                    del self.state.lost_tracks[track_id]
            
            self._previous_track_ids = current_track_ids
            
            return tracks
            
        except Exception as e:
            logger.error(f"Tracking error: {e}")
            return []
    
    def get_lost_track_ids(self) -> List[int]:
        """Get IDs of tracks that were recently lost."""
        return list(self.state.lost_tracks.keys())
    
    def get_active_track_ids(self) -> List[int]:
        """Get IDs of currently active tracks."""
        return list(self._previous_track_ids)
    
    def is_track_active(self, track_id: int) -> bool:
        """Check if a track is currently active."""
        return track_id in self._previous_track_ids
    
    def get_track(self, track_id: int) -> Optional[Track]:
        """Get a specific track by ID."""
        return self.state.active_tracks.get(track_id)
    
    def reset(self):
        """Reset tracker state."""
        if self.tracker:
            self.tracker.reset()
        self.state = TrackerState()
        self._previous_track_ids = set()
        logger.info("Tracker reset")
    
    @property
    def is_initialized(self) -> bool:
        return self._initialized
