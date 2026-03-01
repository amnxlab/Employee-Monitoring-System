"""
Event-based attendance logger with unique IDs, JSON storage, and session
continuity.

Each clock-in starts a **session** identified by an event ID.  Subsequent
TEMP_LOST / RECOVERED / CLOCK_OUT events reference the same parent session
instead of creating new ones.

Storage:  ``data/logs/{emp_id}_{YYYY}_W{WW}.json``
"""

import json
import hashlib
import logging
import os
import random
import string
import time
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

import config
from utils.helpers import format_duration, format_hours, get_week_number

logger = logging.getLogger(__name__)

# ── event ID generation ────────────────────────────────────────────────

def _generate_event_id(dt: datetime = None) -> str:
    """Generate a unique event ID: EVT-YYYYMMDD-HHMMSS-XXXX."""
    if dt is None:
        dt = datetime.now()
    suffix = "".join(random.choices(string.ascii_uppercase + string.digits, k=4))
    return f"EVT-{dt.strftime('%Y%m%d-%H%M%S')}-{suffix}"


# ── data structures ───────────────────────────────────────────────────

@dataclass
class Interruption:
    """A TEMP_LOST -> RECOVERED interval within a session."""
    lost_at: str        # ISO timestamp
    recovered_at: Optional[str] = None   # ISO timestamp, None if still lost
    event_id_lost: str = ""
    event_id_recovered: str = ""


@dataclass
class Session:
    """One continuous work session from CLOCK_IN to CLOCK_OUT."""
    event_id: str                          # the CLOCK_IN event ID
    employee_id: str
    employee_name: str
    clock_in: str                          # ISO timestamp
    clock_out: Optional[str] = None        # ISO timestamp
    duration_seconds: float = 0.0
    date_str: str = ""                     # YYYY-MM-DD
    snapshot_path: Optional[str] = None    # relative path to snapshot
    interruptions: List[Interruption] = field(default_factory=list)
    closed: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["interruptions"] = [asdict(i) for i in self.interruptions]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Session":
        intrs = [Interruption(**i) for i in d.get("interruptions", [])]
        d = dict(d)
        d["interruptions"] = intrs
        return cls(**d)


@dataclass
class WeeklyLog:
    """Full weekly log for one employee."""
    employee_id: str
    employee_name: str
    year: int
    week: int
    sessions: List[Session] = field(default_factory=list)
    daily_totals: Dict[str, float] = field(default_factory=dict)  # date -> secs
    weekly_total_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "employee_id": self.employee_id,
            "employee_name": self.employee_name,
            "year": self.year,
            "week": self.week,
            "sessions": [s.to_dict() for s in self.sessions],
            "daily_totals": self.daily_totals,
            "weekly_total_seconds": self.weekly_total_seconds,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WeeklyLog":
        sessions = [Session.from_dict(s) for s in d.get("sessions", [])]
        return cls(
            employee_id=d["employee_id"],
            employee_name=d["employee_name"],
            year=d["year"],
            week=d["week"],
            sessions=sessions,
            daily_totals=d.get("daily_totals", {}),
            weekly_total_seconds=d.get("weekly_total_seconds", 0.0),
        )


# ── main logger class ─────────────────────────────────────────────────

class EventLogger:
    """
    Event-based attendance logger.

    Usage::

        el = EventLogger()
        eid = el.log_clock_in("EMP005", "The Boss")
        el.log_temp_lost("EMP005", eid)
        el.log_recovered("EMP005", eid)
        el.log_clock_out("EMP005", eid, duration_seconds=3600)
    """

    def __init__(self, logs_dir: Path = None):
        self.logs_dir = logs_dir or config.LOGS_DIR
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache: (emp_id, year, week) -> WeeklyLog
        self._cache: Dict[Tuple[str, int, int], WeeklyLog] = {}

        # emp_id -> current open session event_id
        self._active_sessions: Dict[str, str] = {}

        # Recover orphaned sessions from previous runs
        self._recover_orphaned_sessions()

    # ── orphaned session recovery ──────────────────────────────────────

    def _recover_orphaned_sessions(self):
        """Scan existing JSON logs and close orphaned (unclosed) sessions.

        On startup, any session with ``closed == False`` from a previous run
        can never be closed because the in-memory ``_active_sessions`` mapping
        was lost.  This method:
        1. Restores still-current-week sessions into ``_active_sessions``.
        2. Force-closes sessions from older weeks with a recovery note.
        """
        now = datetime.now()
        current_year = now.year
        current_week = get_week_number(now)
        recovered = 0
        closed = 0

        for path in self.logs_dir.glob("*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                wl = WeeklyLog.from_dict(data)
            except Exception:
                continue

            changed = False
            for session in wl.sessions:
                if session.closed:
                    continue

                is_current_week = (wl.year == current_year and wl.week == current_week)

                if is_current_week:
                    # Restore into active sessions so a future clock-out can close it
                    self._active_sessions[session.employee_id] = session.event_id
                    recovered += 1
                else:
                    # Old week — force-close with the last known timestamp
                    last_ts = session.clock_in  # fallback
                    if session.interruptions:
                        last_intr = session.interruptions[-1]
                        last_ts = last_intr.recovered_at or last_intr.lost_at or last_ts
                    session.clock_out = last_ts
                    session.closed = True
                    session.duration_seconds = 0.0  # unknown — mark as 0
                    # Close any open interruptions
                    for intr in session.interruptions:
                        if intr.recovered_at is None:
                            intr.recovered_at = last_ts
                    changed = True
                    closed += 1

            if changed:
                key = (wl.employee_id, wl.year, wl.week)
                self._cache[key] = wl
                self._save(wl)

        if recovered or closed:
            logger.info(
                f"Session recovery: {recovered} restored to active, "
                f"{closed} force-closed from old weeks"
            )

    # ── file I/O ───────────────────────────────────────────────────────

    def _log_path(self, employee_id: str, year: int, week: int) -> Path:
        return self.logs_dir / f"{employee_id}_{year}_W{week:02d}.json"

    def _load(self, employee_id: str, year: int, week: int) -> WeeklyLog:
        key = (employee_id, year, week)
        if key in self._cache:
            return self._cache[key]

        path = self._log_path(employee_id, year, week)
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                wl = WeeklyLog.from_dict(data)
            except Exception as e:
                logger.error(f"Failed to load log {path}: {e}")
                wl = WeeklyLog(
                    employee_id=employee_id,
                    employee_name="",
                    year=year, week=week
                )
        else:
            wl = WeeklyLog(
                employee_id=employee_id,
                employee_name="",
                year=year, week=week
            )
        self._cache[key] = wl
        # Evict oldest entries if cache grows too large
        while len(self._cache) > 50:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        return wl

    def _save(self, wl: WeeklyLog):
        """Atomic write: write to temp file then rename to prevent corruption."""
        path = self._log_path(wl.employee_id, wl.year, wl.week)
        tmp_path = path.with_suffix(".tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(wl.to_dict(), f, indent=2, ensure_ascii=False)
            # Atomic rename (on Windows, os.replace is atomic for same volume)
            os.replace(str(tmp_path), str(path))
        except Exception as e:
            logger.error(f"Failed to save log {path}: {e}")
            # Cleanup temp file on failure
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    def _current_wl(self, employee_id: str, name: str = "") -> WeeklyLog:
        now = datetime.now()
        wl = self._load(employee_id, now.year, get_week_number(now))
        if name:
            wl.employee_name = name
            # Also update name in any open (unclosed) sessions to keep consistency
            for session in wl.sessions:
                if not session.closed and session.employee_id == employee_id:
                    session.employee_name = name
        return wl

    def _find_session(self, wl: WeeklyLog, event_id: str) -> Optional[Session]:
        for s in wl.sessions:
            if s.event_id == event_id:
                return s
        return None

    # ── public API ─────────────────────────────────────────────────────

    def log_clock_in(
        self,
        employee_id: str,
        employee_name: str,
        snapshot_path: Optional[str] = None,
        timestamp: datetime = None,
    ) -> str:
        """
        Log a CLOCK_IN event.  Creates a new session.

        Returns:
            The generated event ID (used as parent for subsequent events).
        """
        if timestamp is None:
            timestamp = datetime.now()

        event_id = _generate_event_id(timestamp)
        wl = self._current_wl(employee_id, employee_name)

        session = Session(
            event_id=event_id,
            employee_id=employee_id,
            employee_name=employee_name,
            clock_in=timestamp.isoformat(),
            date_str=timestamp.strftime("%Y-%m-%d"),
            snapshot_path=snapshot_path,
        )
        wl.sessions.append(session)
        self._active_sessions[employee_id] = event_id
        self._save(wl)

        logger.info(f"[{event_id}] CLOCK_IN {employee_name} ({employee_id})")
        return event_id

    def update_snapshot_path(self, employee_id: str, event_id: str, snapshot_path: str):
        """Update the snapshot path for an existing session (after snapshot is saved)."""
        wl = self._current_wl(employee_id)
        session = self._find_session(wl, event_id)
        if session:
            session.snapshot_path = snapshot_path
            self._save(wl)

    def log_temp_lost(
        self,
        employee_id: str,
        parent_event_id: str = None,
        timestamp: datetime = None,
    ) -> Optional[str]:
        """Log a TEMP_LOST event under the parent session."""
        if timestamp is None:
            timestamp = datetime.now()

        event_id = _generate_event_id(timestamp)
        parent = parent_event_id or self._active_sessions.get(employee_id)
        if parent is None:
            logger.warning(f"TEMP_LOST for {employee_id} but no active session")
            return None

        wl = self._current_wl(employee_id)
        session = self._find_session(wl, parent)
        if session:
            intr = Interruption(
                lost_at=timestamp.isoformat(),
                event_id_lost=event_id,
            )
            session.interruptions.append(intr)
            self._save(wl)

        logger.info(f"[{event_id}] TEMP_LOST {employee_id} (parent: {parent})")
        return event_id

    def log_recovered(
        self,
        employee_id: str,
        parent_event_id: str = None,
        timestamp: datetime = None,
    ) -> Optional[str]:
        """Log a RECOVERED event under the parent session."""
        if timestamp is None:
            timestamp = datetime.now()

        event_id = _generate_event_id(timestamp)
        parent = parent_event_id or self._active_sessions.get(employee_id)
        if parent is None:
            logger.warning(f"RECOVERED for {employee_id} but no active session")
            return None

        wl = self._current_wl(employee_id)
        session = self._find_session(wl, parent)
        if session and session.interruptions:
            # Close the most recent open interruption
            for intr in reversed(session.interruptions):
                if intr.recovered_at is None:
                    intr.recovered_at = timestamp.isoformat()
                    intr.event_id_recovered = event_id
                    break
            self._save(wl)

        logger.info(f"[{event_id}] RECOVERED {employee_id} (parent: {parent})")
        return event_id

    def log_clock_out(
        self,
        employee_id: str,
        parent_event_id: str = None,
        duration_seconds: float = 0.0,
        timestamp: datetime = None,
    ) -> Optional[str]:
        """Log a CLOCK_OUT event and close the session."""
        if timestamp is None:
            timestamp = datetime.now()

        event_id = _generate_event_id(timestamp)
        parent = parent_event_id or self._active_sessions.get(employee_id)
        if parent is None:
            logger.warning(f"CLOCK_OUT for {employee_id} but no active session")
            return None

        wl = self._current_wl(employee_id)
        session = self._find_session(wl, parent)
        if session:
            session.clock_out = timestamp.isoformat()
            session.duration_seconds = duration_seconds
            session.closed = True

            # Close any open interruptions
            for intr in session.interruptions:
                if intr.recovered_at is None:
                    intr.recovered_at = timestamp.isoformat()

            # Update daily total
            day = session.date_str
            wl.daily_totals[day] = wl.daily_totals.get(day, 0.0) + duration_seconds

            # Update weekly total
            wl.weekly_total_seconds = sum(wl.daily_totals.values())
            self._save(wl)

        self._active_sessions.pop(employee_id, None)

        logger.info(
            f"[{event_id}] CLOCK_OUT {employee_id} (parent: {parent}) "
            f"duration: {format_duration(duration_seconds)}"
        )
        return event_id

    def log_daily_summary(
        self,
        employee_id: str,
        employee_name: str,
        daily_seconds: float = None,
        day: str = None,
    ):
        """Reconcile & persist the daily total in the weekly log.

        If *daily_seconds* is ``None``, the total is computed from closed
        sessions for that day (single source of truth).  Otherwise the
        value is accepted as an override.
        """
        if day is None:
            day = datetime.now().strftime("%Y-%m-%d")
        wl = self._current_wl(employee_id, employee_name)

        if daily_seconds is None:
            # Derive from closed session data — authoritative
            daily_seconds = sum(
                s.duration_seconds
                for s in wl.sessions
                if s.date_str == day and s.closed
            )

        wl.daily_totals[day] = daily_seconds
        wl.weekly_total_seconds = sum(wl.daily_totals.values())
        self._save(wl)

    # ── queries ────────────────────────────────────────────────────────

    def get_active_event_id(self, employee_id: str) -> Optional[str]:
        """Get the event ID of the current open session for an employee."""
        return self._active_sessions.get(employee_id)

    def get_weekly_log(self, employee_id: str, dt: datetime = None) -> Optional[WeeklyLog]:
        if dt is None:
            dt = datetime.now()
        return self._load(employee_id, dt.year, get_week_number(dt))

    def get_daily_total(self, employee_id: str, day: str = None) -> float:
        if day is None:
            day = datetime.now().strftime("%Y-%m-%d")
        wl = self._current_wl(employee_id)
        return wl.daily_totals.get(day, 0.0)

    def get_weekly_total(self, employee_id: str) -> float:
        wl = self._current_wl(employee_id)
        return wl.weekly_total_seconds

    def generate_weekly_report(
        self,
        employee_ids: List[str],
        employee_names: Dict[str, str],
    ) -> str:
        """Generate a human-readable weekly report."""
        lines = ["Weekly Work Report", "-" * 40]
        for emp_id in sorted(employee_ids):
            name = employee_names.get(emp_id, emp_id)
            total = self.get_weekly_total(emp_id)
            hours = total / 3600
            lines.append(f"  {name}: {hours:.1f} hrs")

            # Daily breakdown
            wl = self._current_wl(emp_id, name)
            for day in sorted(wl.daily_totals.keys()):
                dh = wl.daily_totals[day] / 3600
                lines.append(f"    {day}: {dh:.1f} hrs")

        return "\n".join(lines)
