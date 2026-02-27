import logging
import time
from typing import Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime

from utils.helpers import get_week_number, is_new_week

logger = logging.getLogger(__name__)


@dataclass
class EmployeeTime:
    """Time tracking data for a single employee."""
    employee_id: str
    
    current_session_start: Optional[float] = None
    current_session_seconds: float = 0.0
    
    daily_seconds: float = 0.0
    weekly_seconds: float = 0.0
    
    last_update: float = field(default_factory=time.time)
    week_number: int = field(default_factory=lambda: get_week_number())


class TimerManager:
    """
    Manages work time tracking for all employees.
    
    Tracks:
    - Current session duration
    - Daily accumulated time
    - Weekly accumulated time
    
    Handles weekly reset on Monday.
    """
    
    def __init__(self):
        self.employees: Dict[str, EmployeeTime] = {}
        self._last_week_check = datetime.now()
    
    def _get_or_create(self, employee_id: str) -> EmployeeTime:
        """Get or create time tracking for an employee."""
        if employee_id not in self.employees:
            self.employees[employee_id] = EmployeeTime(employee_id=employee_id)
        return self.employees[employee_id]
    
    def clock_in(self, employee_id: str, timestamp: float = None):
        """Record clock-in for an employee."""
        if timestamp is None:
            timestamp = time.time()
        
        emp_time = self._get_or_create(employee_id)
        
        self._check_weekly_reset(emp_time)
        
        emp_time.current_session_start = timestamp
        emp_time.current_session_seconds = 0.0
        emp_time.last_update = timestamp
        
        logger.info(f"Timer started for {employee_id}")
    
    def clock_out(self, employee_id: str, timestamp: float = None, deduct_seconds: float = 0.0):
        """
        Record clock-out for an employee.
        
        Args:
            employee_id: Employee identifier
            timestamp: Clock-out time
            deduct_seconds: Seconds to deduct (e.g., TEMP_LOST timeout)
        """
        if timestamp is None:
            timestamp = time.time()
        
        emp_time = self._get_or_create(employee_id)
        
        if emp_time.current_session_start is not None:
            session_duration = timestamp - emp_time.current_session_start - deduct_seconds
            session_duration = max(0, session_duration)
            
            emp_time.current_session_seconds = session_duration
            emp_time.daily_seconds += session_duration
            emp_time.weekly_seconds += session_duration
            
            logger.info(
                f"Timer stopped for {employee_id}: "
                f"session={session_duration:.0f}s, "
                f"daily={emp_time.daily_seconds:.0f}s, "
                f"weekly={emp_time.weekly_seconds:.0f}s"
            )
        
        emp_time.current_session_start = None
        emp_time.last_update = timestamp
    
    def get_current_session(self, employee_id: str, current_time: float = None) -> float:
        """Get current session duration in seconds."""
        if current_time is None:
            current_time = time.time()
        
        emp_time = self.employees.get(employee_id)
        if emp_time is None or emp_time.current_session_start is None:
            return 0.0
        
        return current_time - emp_time.current_session_start
    
    def get_daily_total(self, employee_id: str) -> float:
        """Get daily total seconds for an employee."""
        emp_time = self.employees.get(employee_id)
        if emp_time is None:
            return 0.0
        return emp_time.daily_seconds
    
    def get_weekly_total(self, employee_id: str) -> float:
        """Get weekly total seconds for an employee."""
        emp_time = self.employees.get(employee_id)
        if emp_time is None:
            return 0.0
        return emp_time.weekly_seconds
    
    def get_last_session_duration(self, employee_id: str) -> float:
        """Get the duration of the last completed session."""
        emp_time = self.employees.get(employee_id)
        if emp_time is None:
            return 0.0
        return emp_time.current_session_seconds
    
    def get_all_weekly_totals(self) -> Dict[str, float]:
        """Get weekly totals for all employees."""
        return {
            emp_id: emp_time.weekly_seconds
            for emp_id, emp_time in self.employees.items()
        }
    
    def _check_weekly_reset(self, emp_time: EmployeeTime):
        """Check and perform weekly reset if needed."""
        current_week = get_week_number()
        
        if emp_time.week_number != current_week:
            logger.info(
                f"Weekly reset for {emp_time.employee_id}: "
                f"week {emp_time.week_number} -> {current_week}"
            )
            emp_time.weekly_seconds = 0.0
            emp_time.week_number = current_week
    
    def reset_daily(self, employee_id: str = None):
        """Reset daily totals for one or all employees."""
        if employee_id:
            if employee_id in self.employees:
                self.employees[employee_id].daily_seconds = 0.0
        else:
            for emp_time in self.employees.values():
                emp_time.daily_seconds = 0.0
        
        logger.info(f"Daily reset: {employee_id or 'all employees'}")
    
    def reset_weekly(self, employee_id: str = None):
        """Reset weekly totals for one or all employees."""
        if employee_id:
            if employee_id in self.employees:
                self.employees[employee_id].weekly_seconds = 0.0
                self.employees[employee_id].week_number = get_week_number()
        else:
            current_week = get_week_number()
            for emp_time in self.employees.values():
                emp_time.weekly_seconds = 0.0
                emp_time.week_number = current_week
        
        logger.info(f"Weekly reset: {employee_id or 'all employees'}")
    
    def is_clocked_in(self, employee_id: str) -> bool:
        """Check if an employee is currently clocked in."""
        emp_time = self.employees.get(employee_id)
        return emp_time is not None and emp_time.current_session_start is not None
