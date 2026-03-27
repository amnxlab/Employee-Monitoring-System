"""
Discord Schedules channel reader.

Reads schedule messages from a dedicated Discord channel using the Bot API,
parses them into per-employee shift entries, and provides a simple query
interface used by MonitoringSystem to determine whether a clock-in is late
or a clock-out is early.

Expected message format (posted by manager in the Schedules channel):
    Week 13 (2026)
    Alice: Mon 09:00-17:00, Tue 09:00-17:00, Wed OFF, Thu 09:00-17:00
    Bob: Mon 10:00-18:00, Fri 10:00-16:00

Rules:
  - The "Week N (YYYY)" header identifies which ISO week the block covers.
  - Employee names are matched case-insensitively; partial match on first name works.
  - "OFF" means no shift that day.
  - Days: Mon Tue Wed Thu Fri Sat Sun  (3-letter abbreviations, case-insensitive)
  - If DISCORD_BOT_TOKEN or DISCORD_SCHEDULES_CHANNEL_ID are not configured,
    all methods return None / False and schedule comparison is silently disabled.
"""

import logging
import re
import threading
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Dict, List, Optional, Tuple

import requests

import config
from utils.helpers import get_week_number

logger = logging.getLogger(__name__)

# Maps 3-letter day abbreviation → Python weekday number (Mon=0 … Sun=6)
_DAY_MAP = {
    "mon": 0, "tue": 1, "wed": 2, "thu": 3,
    "fri": 4, "sat": 5, "sun": 6,
}

_DISCORD_API = "https://discord.com/api/v10"
_FETCH_LIMIT = 50  # messages to fetch per request


@dataclass
class ShiftEntry:
    """A single employee shift."""
    employee_name: str   # as it appears in the schedule message
    shift_date: date
    start: time
    end: time


class ScheduleReader:
    """
    Reads and caches the current week's schedule from a Discord Schedules channel.

    All public methods are safe to call even if the bot is not configured —
    they return None / False, disabling schedule-based comparisons gracefully.
    """

    def __init__(self):
        self._token: str = config.DISCORD_BOT_TOKEN
        self._channel_id: str = config.DISCORD_SCHEDULES_CHANNEL_ID
        self._enabled: bool = bool(self._token and self._channel_id)

        # cache: (year, week) -> list of ShiftEntry
        self._cache: Dict[Tuple[int, int], List[ShiftEntry]] = {}
        self._lock = threading.Lock()
        self._cached_week: Optional[Tuple[int, int]] = None

        if not self._enabled:
            logger.info(
                "ScheduleReader: DISCORD_BOT_TOKEN or DISCORD_SCHEDULES_CHANNEL_ID "
                "not set — schedule comparison disabled"
            )
        else:
            logger.info(
                f"ScheduleReader: enabled (channel {self._channel_id})"
            )

    # ── public API ────────────────────────────────────────────────────

    def refresh(self) -> bool:
        """
        Fetch and (re-)parse the current week's schedule from Discord.
        Safe to call on startup and on week boundary.
        Returns True if a schedule was found and parsed.
        """
        if not self._enabled:
            return False

        year = datetime.now().year
        week = get_week_number()

        shifts = self._fetch_week(year, week)
        with self._lock:
            self._cache[(year, week)] = shifts or []
            self._cached_week = (year, week)

        if shifts:
            logger.info(
                f"ScheduleReader: loaded {len(shifts)} shift(s) for "
                f"{year}-W{week:02d}"
            )
            return True
        else:
            logger.info(
                f"ScheduleReader: no schedule found for {year}-W{week:02d} "
                "— late/early checks disabled this week"
            )
            return False

    def has_schedule_for_week(self, year: int, week: int) -> bool:
        """True if at least one shift is cached for the given week."""
        with self._lock:
            return bool(self._cache.get((year, week)))

    def get_shift(self, employee_name: str, shift_date: date) -> Optional[ShiftEntry]:
        """
        Return the ShiftEntry for an employee on a specific date, or None.

        employee_name — the name from the face recognition database (e.g. "Alice Smith").
        Matching is case-insensitive; the first word (first name) is enough.
        """
        year, week, _ = shift_date.isocalendar()
        with self._lock:
            shifts = self._cache.get((year, week), [])

        if not shifts:
            return None

        first_name = employee_name.strip().split()[0].lower()
        for entry in shifts:
            entry_first = entry.employee_name.strip().split()[0].lower()
            if entry_first == first_name and entry.shift_date == shift_date:
                return entry
        return None

    # ── private helpers ────────────────────────────────────────────────

    def _headers(self) -> dict:
        return {"Authorization": f"Bot {self._token}"}

    def _fetch_week(self, year: int, week: int) -> Optional[List[ShiftEntry]]:
        """Fetch recent channel messages and parse the schedule block for the given week."""
        try:
            resp = requests.get(
                f"{_DISCORD_API}/channels/{self._channel_id}/messages",
                headers=self._headers(),
                params={"limit": _FETCH_LIMIT},
                timeout=10,
            )
            if resp.status_code == 401:
                logger.error("ScheduleReader: invalid bot token (401)")
                return None
            if resp.status_code == 403:
                logger.error(
                    "ScheduleReader: bot lacks Read Message History permission (403)"
                )
                return None
            if resp.status_code != 200:
                logger.error(
                    f"ScheduleReader: unexpected status {resp.status_code}"
                )
                return None

            messages = resp.json()
        except requests.exceptions.Timeout:
            logger.error("ScheduleReader: request timed out")
            return None
        except Exception as exc:
            logger.error(f"ScheduleReader: fetch error — {exc}")
            return None

        # Discord returns messages newest-first; search all of them
        for msg in messages:
            content: str = msg.get("content", "")
            shifts = self._parse_schedule_block(content, year, week)
            if shifts is not None:
                return shifts

        return None

    # ── parser ────────────────────────────────────────────────────────

    # Pattern: "Week 13 (2026)" or "week13 2026" or "W13 2026" etc.
    _WEEK_HDR = re.compile(
        r"(?:week|w)\s*(\d{1,2})\s*[\(\s]\s*(\d{4})\s*\)?",
        re.IGNORECASE,
    )

    # Pattern: "Alice: Mon 09:00-17:00, Tue 10:00-18:00, Wed OFF"
    _EMP_LINE = re.compile(r"^(.+?)\s*:\s*(.+)$", re.MULTILINE)

    # Pattern for a single day entry: "Mon 09:00-17:00" or "Mon OFF"
    _DAY_ENTRY = re.compile(
        r"(mon|tue|wed|thu|fri|sat|sun)\s+"
        r"(?:(off)|(\d{1,2}:\d{2})\s*[-–]\s*(\d{1,2}:\d{2}))",
        re.IGNORECASE,
    )

    def _parse_schedule_block(
        self,
        text: str,
        target_year: int,
        target_week: int,
    ) -> Optional[List[ShiftEntry]]:
        """
        Parse a single Discord message.  Returns a list of ShiftEntry objects
        if the message contains a schedule header matching (target_year, target_week),
        or None if this message is not a schedule for that week.
        """
        hdr_match = self._WEEK_HDR.search(text)
        if not hdr_match:
            return None

        msg_week = int(hdr_match.group(1))
        msg_year = int(hdr_match.group(2))
        if msg_week != target_week or msg_year != target_year:
            return None

        # Find the Monday of that ISO week so we can map day names to dates
        monday = _iso_week_monday(target_year, target_week)

        shifts: List[ShiftEntry] = []

        for emp_match in self._EMP_LINE.finditer(text):
            emp_name = emp_match.group(1).strip()
            day_str = emp_match.group(2)

            for day_match in self._DAY_ENTRY.finditer(day_str):
                day_abbr = day_match.group(1).lower()
                is_off = day_match.group(2) is not None
                if is_off:
                    continue  # no shift entry for OFF days

                start_str = day_match.group(3)
                end_str = day_match.group(4)

                weekday_offset = _DAY_MAP.get(day_abbr)
                if weekday_offset is None:
                    continue

                from datetime import timedelta
                shift_date = monday + timedelta(days=weekday_offset)
                start_t = _parse_time(start_str)
                end_t = _parse_time(end_str)
                if start_t is None or end_t is None:
                    continue

                shifts.append(ShiftEntry(
                    employee_name=emp_name,
                    shift_date=shift_date,
                    start=start_t,
                    end=end_t,
                ))

        return shifts  # may be empty list (week header found but no valid lines)


# ── helpers ───────────────────────────────────────────────────────────────


def _iso_week_monday(year: int, week: int) -> date:
    """Return the Monday date of the given ISO week."""
    # ISO week 1 of year Y is the week containing the first Thursday of Y.
    # Python's date.fromisocalendar is available from 3.8+.
    return date.fromisocalendar(year, week, 1)


def _parse_time(s: str) -> Optional[time]:
    """Parse 'HH:MM' or 'H:MM' string into a datetime.time object."""
    try:
        parts = s.strip().split(":")
        return time(int(parts[0]), int(parts[1]))
    except Exception:
        return None
