import logging
import time
from enum import Enum
from typing import Optional, Callable
from dataclasses import dataclass, field

import config

logger = logging.getLogger(__name__)


class AttendanceState(Enum):
    OUT = "out"
    DETECTED = "detected"
    CLOCKED_IN = "clocked_in"
    TEMP_LOST = "temp_lost"
    CLOCKED_OUT = "clocked_out"


@dataclass
class StateTransition:
    """Represents a state transition event."""
    from_state: AttendanceState
    to_state: AttendanceState
    timestamp: float
    employee_id: str


@dataclass
class AttendanceStateMachineState:
    """Internal state for the state machine."""
    current_state: AttendanceState = AttendanceState.OUT
    
    clock_in_time: Optional[float] = None
    clock_out_time: Optional[float] = None
    temp_lost_time: Optional[float] = None
    
    visible_frame_count: int = 0
    invisible_frame_count: int = 0
    
    session_seconds: float = 0.0
    
    just_clocked_in: bool = False
    just_clocked_out: bool = False
    just_temp_lost: bool = False


class AttendanceStateMachine:
    """
    Per-employee attendance state machine.
    
    States:
    - OUT: Employee not detected
    - DETECTED: Face recognized (transitions immediately to CLOCKED_IN)
    - CLOCKED_IN: Employee is working, timer running
    - TEMP_LOST: Tracking lost, 10-minute countdown active
    - CLOCKED_OUT: Session ended, timer stopped
    
    Implements debounce logic to prevent rapid state changes.
    """
    
    def __init__(
        self,
        employee_id: str,
        temp_lost_timeout: float = None,
        debounce_frames: int = None
    ):
        self.employee_id = employee_id
        self.temp_lost_timeout = temp_lost_timeout or config.TEMP_LOST_TIMEOUT_SECONDS
        self.debounce_frames = debounce_frames or config.DEBOUNCE_FRAMES
        
        self.state = AttendanceStateMachineState()
        self._transition_callbacks: list = []
    
    def update(self, is_visible: bool, current_time: float = None) -> Optional[StateTransition]:
        """
        Update state machine based on visibility.
        
        Args:
            is_visible: Whether the employee is currently visible
            current_time: Current timestamp (defaults to time.time())
            
        Returns:
            StateTransition if a transition occurred, None otherwise
        """
        if current_time is None:
            current_time = time.time()
        
        self.state.just_clocked_in = False
        self.state.just_clocked_out = False
        self.state.just_temp_lost = False
        
        if is_visible:
            self.state.visible_frame_count += 1
            self.state.invisible_frame_count = 0
        else:
            self.state.invisible_frame_count += 1
            self.state.visible_frame_count = 0
        
        transition = None
        current = self.state.current_state
        
        if current == AttendanceState.OUT:
            transition = self._handle_out_state(is_visible, current_time)
        
        elif current == AttendanceState.DETECTED:
            transition = self._handle_detected_state(current_time)
        
        elif current == AttendanceState.CLOCKED_IN:
            transition = self._handle_clocked_in_state(is_visible, current_time)
        
        elif current == AttendanceState.TEMP_LOST:
            transition = self._handle_temp_lost_state(is_visible, current_time)
        
        elif current == AttendanceState.CLOCKED_OUT:
            transition = self._handle_clocked_out_state(is_visible, current_time)
        
        if transition:
            self._notify_transition(transition)
        
        return transition
    
    def _handle_out_state(self, is_visible: bool, current_time: float) -> Optional[StateTransition]:
        """Handle OUT state transitions."""
        if is_visible and self.state.visible_frame_count >= self.debounce_frames:
            return self._transition_to(AttendanceState.DETECTED, current_time)
        return None
    
    def _handle_detected_state(self, current_time: float) -> Optional[StateTransition]:
        """Handle DETECTED state - immediately transition to CLOCKED_IN."""
        self.state.clock_in_time = current_time
        self.state.session_seconds = 0.0
        self.state.just_clocked_in = True
        
        return self._transition_to(AttendanceState.CLOCKED_IN, current_time)
    
    def _handle_clocked_in_state(self, is_visible: bool, current_time: float) -> Optional[StateTransition]:
        """Handle CLOCKED_IN state transitions."""
        if not is_visible and self.state.invisible_frame_count >= self.debounce_frames:
            self.state.temp_lost_time = current_time
            self.state.just_temp_lost = True
            return self._transition_to(AttendanceState.TEMP_LOST, current_time)
        return None
    
    def _handle_temp_lost_state(self, is_visible: bool, current_time: float) -> Optional[StateTransition]:
        """Handle TEMP_LOST state transitions."""
        if is_visible and self.state.visible_frame_count >= self.debounce_frames:
            self.state.temp_lost_time = None
            return self._transition_to(AttendanceState.CLOCKED_IN, current_time)
        
        if self.state.temp_lost_time is not None:
            elapsed = current_time - self.state.temp_lost_time
            if elapsed >= self.temp_lost_timeout:
                self.state.clock_out_time = current_time
                
                if self.state.clock_in_time:
                    self.state.session_seconds = (
                        current_time - self.state.clock_in_time - self.temp_lost_timeout
                    )
                
                self.state.just_clocked_out = True
                self.state.temp_lost_time = None
                
                return self._transition_to(AttendanceState.CLOCKED_OUT, current_time)
        
        return None
    
    def _handle_clocked_out_state(self, is_visible: bool, current_time: float) -> Optional[StateTransition]:
        """Handle CLOCKED_OUT state transitions."""
        if is_visible and self.state.visible_frame_count >= self.debounce_frames:
            return self._transition_to(AttendanceState.DETECTED, current_time)
        return None
    
    def _transition_to(self, new_state: AttendanceState, current_time: float) -> StateTransition:
        """Execute state transition."""
        old_state = self.state.current_state
        self.state.current_state = new_state
        
        transition = StateTransition(
            from_state=old_state,
            to_state=new_state,
            timestamp=current_time,
            employee_id=self.employee_id
        )
        
        logger.info(
            f"Employee {self.employee_id}: {old_state.value} -> {new_state.value}"
        )
        
        return transition
    
    def add_transition_callback(self, callback: Callable[[StateTransition], None]):
        """Add a callback to be notified of state transitions."""
        self._transition_callbacks.append(callback)
    
    def _notify_transition(self, transition: StateTransition):
        """Notify all callbacks of a state transition."""
        for callback in self._transition_callbacks:
            try:
                callback(transition)
            except Exception as e:
                logger.error(f"Transition callback error: {e}")
    
    def get_session_duration(self, current_time: float = None) -> float:
        """
        Get current session duration in seconds.
        
        For CLOCKED_IN state, returns time since clock-in.
        For TEMP_LOST state, returns time since clock-in minus lost time.
        For CLOCKED_OUT state, returns final session duration.
        """
        if current_time is None:
            current_time = time.time()
        
        if self.state.current_state == AttendanceState.CLOCKED_IN:
            if self.state.clock_in_time:
                return current_time - self.state.clock_in_time
        
        elif self.state.current_state == AttendanceState.TEMP_LOST:
            if self.state.clock_in_time and self.state.temp_lost_time:
                return self.state.temp_lost_time - self.state.clock_in_time
        
        elif self.state.current_state == AttendanceState.CLOCKED_OUT:
            return self.state.session_seconds
        
        return 0.0
    
    def get_temp_lost_remaining(self, current_time: float = None) -> Optional[float]:
        """Get remaining time before auto clock-out (only in TEMP_LOST state)."""
        if self.state.current_state != AttendanceState.TEMP_LOST:
            return None
        
        if current_time is None:
            current_time = time.time()
        
        if self.state.temp_lost_time:
            elapsed = current_time - self.state.temp_lost_time
            return max(0, self.temp_lost_timeout - elapsed)
        
        return None
    
    def force_clock_out(self, current_time: float = None):
        """Force clock out (e.g., end of day)."""
        if current_time is None:
            current_time = time.time()
        
        if self.state.current_state in [AttendanceState.CLOCKED_IN, AttendanceState.TEMP_LOST]:
            self.state.clock_out_time = current_time
            
            if self.state.clock_in_time:
                if self.state.current_state == AttendanceState.TEMP_LOST and self.state.temp_lost_time:
                    self.state.session_seconds = self.state.temp_lost_time - self.state.clock_in_time
                else:
                    self.state.session_seconds = current_time - self.state.clock_in_time
            
            self.state.just_clocked_out = True
            self._transition_to(AttendanceState.CLOCKED_OUT, current_time)
    
    def reset(self):
        """Reset state machine to initial state."""
        self.state = AttendanceStateMachineState()
        logger.info(f"State machine reset for employee {self.employee_id}")
    
    @property
    def current_state(self) -> AttendanceState:
        return self.state.current_state
    
    @property
    def is_working(self) -> bool:
        """Check if employee is currently working (clocked in or temp lost)."""
        return self.state.current_state in [
            AttendanceState.CLOCKED_IN,
            AttendanceState.TEMP_LOST
        ]
    
    @property
    def clock_in_time(self) -> Optional[float]:
        return self.state.clock_in_time
    
    @property
    def clock_out_time(self) -> Optional[float]:
        return self.state.clock_out_time
