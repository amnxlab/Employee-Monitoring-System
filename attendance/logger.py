import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import config
from utils.helpers import format_duration, format_hours, get_week_number

logger = logging.getLogger(__name__)


class AttendanceLogger:
    """
    Manages attendance log files.
    
    Creates weekly log files per employee in format:
    {employeeID}_{YYYY}_W{weeknum}.log
    
    Log entries include:
    - Clock in/out timestamps
    - Session durations
    - Daily summaries
    """
    
    def __init__(self, logs_dir: Path = None):
        self.logs_dir = logs_dir or config.LOGS_DIR
        self.logs_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_log_path(self, employee_id: str, dt: datetime = None) -> Path:
        """Get log file path for an employee and date."""
        if dt is None:
            dt = datetime.now()
        
        year = dt.year
        week = get_week_number(dt)
        
        filename = f"{employee_id}_{year}_W{week:02d}.log"
        return self.logs_dir / filename
    
    def _write_entry(self, employee_id: str, entry: str):
        """Write an entry to the employee's log file."""
        log_path = self._get_log_path(employee_id)
        
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(entry + "\n")
        except Exception as e:
            logger.error(f"Failed to write log entry: {e}")
    
    def log_clock_in(self, employee_id: str, timestamp: datetime = None):
        """Log a clock-in event."""
        if timestamp is None:
            timestamp = datetime.now()
        
        entry = f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')} | CLOCK_IN"
        self._write_entry(employee_id, entry)
        
        logger.debug(f"Logged clock-in for {employee_id}")
    
    def log_clock_out(
        self,
        employee_id: str,
        timestamp: datetime = None,
        duration_seconds: float = 0.0
    ):
        """Log a clock-out event with session duration."""
        if timestamp is None:
            timestamp = datetime.now()
        
        duration_str = format_duration(duration_seconds)
        entry = f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')} | CLOCK_OUT | duration: {duration_str}"
        self._write_entry(employee_id, entry)
        
        logger.debug(f"Logged clock-out for {employee_id}: {duration_str}")
    
    def log_temp_lost(self, employee_id: str, timestamp: datetime = None):
        """Log when tracking is temporarily lost."""
        if timestamp is None:
            timestamp = datetime.now()
        
        entry = f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')} | TEMP_LOST"
        self._write_entry(employee_id, entry)
    
    def log_recovered(self, employee_id: str, timestamp: datetime = None):
        """Log when tracking is recovered from TEMP_LOST."""
        if timestamp is None:
            timestamp = datetime.now()
        
        entry = f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')} | RECOVERED"
        self._write_entry(employee_id, entry)
    
    def log_daily_summary(
        self,
        employee_id: str,
        daily_seconds: float,
        timestamp: datetime = None
    ):
        """Log daily summary."""
        if timestamp is None:
            timestamp = datetime.now()
        
        hours_str = format_hours(daily_seconds)
        entry = f"{timestamp.strftime('%Y-%m-%d')} | DAILY_TOTAL | {hours_str}"
        self._write_entry(employee_id, entry)
    
    def log_weekly_summary(
        self,
        employee_id: str,
        weekly_seconds: float,
        timestamp: datetime = None
    ):
        """Log weekly summary."""
        if timestamp is None:
            timestamp = datetime.now()
        
        hours_str = format_hours(weekly_seconds)
        entry = f"--- WEEKLY TOTAL: {hours_str} ---"
        self._write_entry(employee_id, entry)
    
    def get_weekly_log(self, employee_id: str, dt: datetime = None) -> Optional[str]:
        """Read the weekly log file for an employee."""
        log_path = self._get_log_path(employee_id, dt)
        
        if not log_path.exists():
            return None
        
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read log file: {e}")
            return None
    
    def get_all_log_files(self, employee_id: str = None) -> List[Path]:
        """Get all log files, optionally filtered by employee."""
        pattern = f"{employee_id}_*.log" if employee_id else "*.log"
        return sorted(self.logs_dir.glob(pattern))
    
    def generate_weekly_report(
        self,
        weekly_totals: Dict[str, float],
        employee_names: Dict[str, str]
    ) -> str:
        """
        Generate a formatted weekly report.
        
        Args:
            weekly_totals: Dict of employee_id -> total seconds
            employee_names: Dict of employee_id -> name
            
        Returns:
            Formatted report string
        """
        lines = ["Weekly Work Report", "-" * 30]
        
        sorted_employees = sorted(
            weekly_totals.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for emp_id, seconds in sorted_employees:
            name = employee_names.get(emp_id, emp_id)
            hours = seconds / 3600
            lines.append(f"- {name}: {hours:.1f} hrs")
        
        return "\n".join(lines)
