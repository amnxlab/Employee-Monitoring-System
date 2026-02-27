"""
Discord webhook integration — 3-channel architecture.

Channels:
  #here-gone       Real-time presence (arrived, left, temp-lost, recovered)
  #clock-in-logs   Formal attendance records, snapshots, daily/weekly reports
  #admin           Security alerts, system events, errors, unknown persons

Each channel has its own webhook URL.  If only the legacy single URL is
configured, all events are sent there.
"""

import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import random
import requests

import config
from utils.helpers import format_duration, format_hours

logger = logging.getLogger(__name__)

# Channel constants
CH_HERE_GONE = "here_gone"
CH_CLOCK_LOGS = "clock_logs"
CH_ADMIN = "admin"


class DiscordNotifier:
    """
    3-channel Discord webhook notifier with rich embeds and file uploads.

    Falls back to single webhook if only DISCORD_WEBHOOK_URL is set.
    """

    def __init__(self):
        # Resolve per-channel webhook URLs
        fallback = config.DISCORD_WEBHOOK_URL
        self._webhooks = {
            CH_HERE_GONE: config.DISCORD_WEBHOOK_HERE_GONE or fallback,
            CH_CLOCK_LOGS: config.DISCORD_WEBHOOK_CLOCK_LOGS or fallback,
            CH_ADMIN: config.DISCORD_WEBHOOK_ADMIN or fallback,
        }
        self._enabled = any(bool(url) for url in self._webhooks.values())

        if not self._enabled:
            logger.warning("No Discord webhooks configured, notifications disabled")
        else:
            active = [k for k, v in self._webhooks.items() if v]
            logger.info(f"Discord channels active: {', '.join(active)}")

    # ═══════════════════════════════════════════════════════════════════
    #  Low-level sending
    # ═══════════════════════════════════════════════════════════════════

    def _send(
        self,
        channel: str,
        content: str = "",
        embed: Optional[dict] = None,
        file_path: Optional[str] = None,
    ) -> bool:
        """Send a message to a specific channel (non-blocking)."""
        url = self._webhooks.get(channel, "")
        if not url:
            return False

        thread = threading.Thread(
            target=self._send_sync,
            args=(url, content, embed, file_path),
            daemon=True,
        )
        thread.start()
        return True

    def _send_sync(
        self,
        webhook_url: str,
        content: str,
        embed: Optional[dict],
        file_path: Optional[str],
    ):
        """Synchronous send (runs on background thread)."""
        try:
            if file_path and Path(file_path).exists():
                # Multipart upload with file
                payload = {}
                if content:
                    payload["content"] = content
                if embed:
                    import json
                    payload["payload_json"] = json.dumps({"embeds": [embed]})

                with open(file_path, "rb") as f:
                    files = {"file": (Path(file_path).name, f, "image/jpeg")}
                    resp = requests.post(
                        webhook_url, data=payload, files=files, timeout=15
                    )
            else:
                # JSON payload (no file)
                payload = {}
                if content:
                    payload["content"] = content
                if embed:
                    payload["embeds"] = [embed]

                resp = requests.post(webhook_url, json=payload, timeout=10)

            if resp.status_code in (200, 204):
                logger.debug(f"Discord sent: {content[:60]}...")
            else:
                logger.error(
                    f"Discord error {resp.status_code}: {resp.text[:200]}"
                )

        except requests.exceptions.Timeout:
            logger.error("Discord webhook timeout")
        except Exception as e:
            logger.error(f"Discord webhook error: {e}")

    # ═══════════════════════════════════════════════════════════════════
    #  Embed builder
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _embed(
        title: str,
        description: str = "",
        color: int = 0x3498DB,
        fields: Optional[list] = None,
        footer: str = "",
        thumbnail_url: str = "",
    ) -> dict:
        """Build a Discord embed dict."""
        e = {"title": title, "color": color}
        if description:
            e["description"] = description
        if fields:
            e["fields"] = fields
        if footer:
            e["footer"] = {"text": footer}
        if thumbnail_url:
            e["thumbnail"] = {"url": thumbnail_url}
        e["timestamp"] = datetime.utcnow().isoformat()
        return e

    @staticmethod
    def _field(name: str, value: str, inline: bool = True) -> dict:
        return {"name": name, "value": value, "inline": inline}

    # ── funny Big Brother message pools (meme edition) ────────────────

    _ARRIVE_MSGS = [
        "A wild employee has appeared! *throws Pokéball*",
        "You shall not pass... without clocking in! 🧙",
        "Somebody once told me this employee was gonna show up 🎵",
        "It's over 9000! ...oh wait, that's just the enthusiasm level",
        "One does not simply walk into the office without Big Brother noticing 👁️",
        "I see you. I always see you. *adjusts cameras menacingly*",
        "Achievement Unlocked: 🏆 Showed Up To Work",
        "Task failed successfully: arrived at work instead of staying in bed",
        "Employee.exe has started running ▶️",
        "This is fine. 🔥🐕 Everything is fine. They're here.",
        "Ah yes, the negotiator. Here to negotiate with the coffee machine first?",
        "Not the hero we deserved, but the one who showed up",
        "Loading enthusiasm... ████████ 100% done!",
        "They came. They saw. They clocked in. 🏛️",
    ]

    _ARRIVE_LATE_MSGS = [
        "Ah, I see you chose the 'fashionably late' DLC 💅",
        "You're late. Gandalf would be disappointed 🧙",
        "Sneak 100. But my cameras have Perception 200 👁️",
        "This is fine. 🔥🐕 You're only *checks notes* ...very late",
        "I've been expecting you, Mr. Bond. For quite a while now ⏰",
        "Error 408: Employee Request Timeout",
        "*surprised Pikachu face* 😮 You're... late?!",
        "Ah yes, the floor here is made of floor. And the late person is late.",
        "You know the rules and so do I. 🎵 Rule #1: Show up on time.",
        "Your alarm clock: 'Am I a joke to you?' 😤",
        "The council will decide your fate 🪑",
        "Did your bed cast a level 99 Sleep spell on you? 💤",
    ]

    _LEAVE_MSGS = [
        "Aight, imma head out 🚶 — so did they apparently",
        "Another one bites the dust! 🎵 *Queen plays*",
        "*sad camera noises* They grow up so fast 😢",
        "To be continued... ➡️ 🎵",
        "I used to be an employee like you, until I took a clock-out to the knee 🏹",
        "It's a terrible day for rain 🌧️ ...oh wait, they just left",
        "You were the chosen one! You were supposed to stay! 😤",
        "Press F to pay respects. F.",
        "Task Manager: Employee.exe has stopped responding ⬜",
        "And they rode off into the sunset 🌅 ...or the parking lot",
        "Goodbye stranger, it's been nice. Hope you find your parking space 🎵",
        "They didn't just leave. They yeeted themselves out. 🏃💨",
    ]

    _LEAVE_EARLY_MSGS = [
        "Sneak 100 🥷 But you can't sneak past Big Brother!",
        "Mission Impossible: Early Escape Protocol activated 🕶️",
        "Speed. I am speed. 🏎️ — This employee, apparently",
        "You underestimate my power! ...to track early exits 😈",
        "We will watch your career with great interest 🪑 ...if you stay longer",
        "Is this an early exit? 🦋 *confused butterfly meme*",
        "Reality can be whatever I want 🟣 ...and they chose 'go home early'",
        "Stealth level: -100. I literally have cameras everywhere.",
    ]

    _LOST_MSGS = [
        "Where banana? 🦧 Where employee??",
        "They vanished like my dad going to get milk 🥛",
        "404: Employee not found 🔍",
        "I guess they chose violence... against the attendance sheet",
        "Hello? Is it me you're looking for? 🎵 Because I can't find YOU",
        "They pulled a Houdini! 🎩 Countdown started...",
        "My disappointment is immeasurable and my cameras are confused 📷",
        "Gone. Reduced to atoms. ⚛️",
        "Invisible mode: ON. But for how long? ⏳",
    ]

    _RECOVERED_MSGS = [
        "The King in the North has returned! 👑🐺",
        "They respawned! 🎮 Welcome back to the server",
        "The comeback kid strikes again 💪",
        "Back from the shadow realm ✨ Yu-Gi-Oh style",
        "Object Identified. Employee.exe is back online 🟢",
        "Guess who's back, back again 🎵",
        "Plot armor activated — they've returned! 🛡️",
        "This isn't even my final form! ...wait, they just came back from the bathroom",
    ]

    # ═══════════════════════════════════════════════════════════════════
    #  Presence events → #here-gone (Big Brother)
    # ═══════════════════════════════════════════════════════════════════

    def send_clock_in(
        self,
        name: str,
        timestamp: Optional[datetime] = None,
        snapshot_path: Optional[str] = None,
        is_late: bool = False,
    ) -> bool:
        ts = timestamp or datetime.now()
        time_str = ts.strftime("%H:%M")

        # #here-gone: funny presence update
        color = config.DISCORD_COLOR_ORANGE if is_late else config.DISCORD_COLOR_GREEN
        if is_late:
            title = f"⏰ {name} arrived (fashionably late)"
            desc = f"**{time_str}** — {random.choice(self._ARRIVE_LATE_MSGS)}"
        else:
            title = f"🟢 {name} arrived"
            desc = f"**{time_str}** — {random.choice(self._ARRIVE_MSGS)}"
        self._send(CH_HERE_GONE, embed=self._embed(title, desc, color))

        # #clock-in-logs: formal record with snapshot
        log_embed = self._embed(
            f"Clock In: {name}",
            f"Time: {time_str}",
            color,
            fields=[
                self._field("Status", "Late" if is_late else "On Time"),
                self._field("Date", ts.strftime("%Y-%m-%d")),
            ],
        )
        self._send(CH_CLOCK_LOGS, embed=log_embed, file_path=snapshot_path)

        # Late arrival is an office matter, not a security concern — no admin alert

        return True

    def send_clock_out(
        self,
        name: str,
        duration_seconds: float,
        timestamp: Optional[datetime] = None,
        is_early: bool = False,
    ) -> bool:
        ts = timestamp or datetime.now()
        time_str = ts.strftime("%H:%M")
        dur_str = format_duration(duration_seconds)
        hours = duration_seconds / 3600

        # #here-gone: funny farewell
        color = config.DISCORD_COLOR_ORANGE if is_early else config.DISCORD_COLOR_RED
        if is_early:
            emoji = "🏃"
            desc = f"**{time_str}** — {random.choice(self._LEAVE_EARLY_MSGS)}\nSession: {dur_str}"
        else:
            emoji = "🔴"
            desc = f"**{time_str}** — {random.choice(self._LEAVE_MSGS)}\nSession: {dur_str}"
        self._send(
            CH_HERE_GONE,
            embed=self._embed(f"{emoji} {name} left", desc, color),
        )

        # #clock-in-logs
        self._send(
            CH_CLOCK_LOGS,
            embed=self._embed(
                f"Clock Out: {name}",
                f"Time: {time_str}",
                color,
                fields=[
                    self._field("Session", dur_str),
                    self._field("Hours", f"{hours:.1f}h"),
                    self._field("Status", "Early" if is_early else "Normal"),
                ],
            ),
        )

        return True

    def send_temp_lost(self, name: str) -> bool:
        # Not sent to Big Brother — only clock in/out go there
        return True

    def send_recovered(self, name: str) -> bool:
        # Not sent to Big Brother — only clock in/out go there
        return True

    # ═══════════════════════════════════════════════════════════════════
    #  Reports → #clock-in-logs
    # ═══════════════════════════════════════════════════════════════════

    def send_daily_summary(self, report_text: str, date: str = "") -> bool:
        """Send daily summary to clock-in-logs."""
        self._send(
            CH_CLOCK_LOGS,
            embed=self._embed(
                f"📊 Daily Summary — {date or datetime.now().strftime('%Y-%m-%d')}",
                report_text,
                config.DISCORD_COLOR_BLUE,
            ),
        )
        return True

    def send_weekly_summary(
        self,
        weekly_totals: Dict[str, float],
        employee_names: Dict[str, str],
    ) -> bool:
        lines = []
        sorted_emps = sorted(weekly_totals.items(), key=lambda x: x[1], reverse=True)
        for emp_id, seconds in sorted_emps:
            name = employee_names.get(emp_id, emp_id)
            hours = seconds / 3600
            bar = "█" * int(hours / 2) + "░" * max(0, 20 - int(hours / 2))
            lines.append(f"`{bar}` **{name}**: {hours:.1f}h")

        if not lines:
            lines.append("_No attendance data this week._")

        self._send(
            CH_CLOCK_LOGS,
            embed=self._embed(
                "📋 Weekly Work Report",
                "\n".join(lines),
                config.DISCORD_COLOR_BLUE,
            ),
        )
        return True

    def send_message(self, text: str) -> bool:
        """Generic message to clock-in-logs (used by daily summary)."""
        self._send(CH_CLOCK_LOGS, content=text)
        return True

    # ═══════════════════════════════════════════════════════════════════
    #  Security & Admin → #admin
    # ═══════════════════════════════════════════════════════════════════

    def send_unknown_person(
        self,
        cam_id: int = 0,
        first_snapshot: Optional[str] = None,
        last_snapshot: Optional[str] = None,
        duration_minutes: float = 0.0,
    ) -> bool:
        """Unknown person alert with first + last snapshots to admin."""
        dur_str = f"{duration_minutes:.0f} min" if duration_minutes else "unknown"
        desc = (
            f"🚨 Unidentified person on **Camera {cam_id}**\n"
            f"Present for **{dur_str}** without face recognition match.\n"
            f"Not matching any registered employee."
        )
        # First snapshot — when they were first seen
        self._send(
            CH_ADMIN,
            embed=self._embed(
                "🔍 Unknown Person — Security Alert",
                desc,
                config.DISCORD_COLOR_RED,
                fields=[
                    self._field("Camera", str(cam_id)),
                    self._field("Duration", dur_str),
                    self._field("First Seen", datetime.now().strftime("%H:%M:%S")),
                ],
            ),
            file_path=first_snapshot,
        )
        # Last snapshot — most recent appearance
        if last_snapshot:
            self._send(
                CH_ADMIN,
                embed=self._embed(
                    "📸 Latest Snapshot",
                    f"Most recent capture of unidentified person on Camera {cam_id}",
                    config.DISCORD_COLOR_PURPLE,
                ),
                file_path=last_snapshot,
            )
        return True

    def send_wrong_password(self) -> bool:
        """Failed shutdown password attempt."""
        self._send(
            CH_ADMIN,
            embed=self._embed(
                "🚨 Failed Shutdown Attempt",
                f"Someone entered a **wrong password** at "
                f"{datetime.now().strftime('%H:%M:%S')}.\n"
                "The system rejected the shutdown request.",
                config.DISCORD_COLOR_RED,
            ),
        )
        return True

    def send_tamper_alert(self, cam_id: int, reason: str = "covered or frozen") -> bool:
        """Camera tamper detection."""
        self._send(
            CH_ADMIN,
            embed=self._embed(
                f"⚠️ Camera {cam_id} Tamper Alert",
                f"Camera {cam_id} appears to be **{reason}**.\n"
                "Please check the physical camera.",
                config.DISCORD_COLOR_RED,
                fields=[
                    self._field("Camera", str(cam_id)),
                    self._field("Time", datetime.now().strftime("%H:%M:%S")),
                ],
            ),
        )
        return True

    def send_overtime_alert(self, name: str, hours: float) -> bool:
        """Employee working overtime."""
        self._send(
            CH_ADMIN,
            embed=self._embed(
                f"⏰ Overtime: {name}",
                f"**{name}** has been working for **{hours:.1f} hours** today.",
                config.DISCORD_COLOR_ORANGE,
            ),
        )
        # Also notify here-gone
        self._send(
            CH_HERE_GONE,
            embed=self._embed(
                f"⏰ {name} in overtime",
                f"Working {hours:.1f}h today",
                config.DISCORD_COLOR_ORANGE,
            ),
        )
        return True

    def send_late_arrival(self, name: str, actual_time: str, expected_hour: int) -> bool:
        """Late arrival notification to admin."""
        self._send(
            CH_ADMIN,
            embed=self._embed(
                f"⏰ Late Arrival: {name}",
                f"Arrived at **{actual_time}** (expected {expected_hour}:00)",
                config.DISCORD_COLOR_ORANGE,
            ),
        )
        return True

    def send_watchdog_restart(self, exit_code: int) -> bool:
        """Watchdog restarted the main process."""
        self._send(
            CH_ADMIN,
            embed=self._embed(
                "🔄 System Auto-Restarted",
                f"The watchdog detected a crash (exit code {exit_code}) "
                f"and restarted the monitoring system.",
                config.DISCORD_COLOR_RED,
                fields=[
                    self._field("Exit Code", str(exit_code)),
                    self._field("Time", datetime.now().strftime("%H:%M:%S")),
                ],
            ),
        )
        return True

    # ═══════════════════════════════════════════════════════════════════
    #  System events → #admin
    # ═══════════════════════════════════════════════════════════════════

    def send_system_start(self) -> bool:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._send(
            CH_ADMIN,
            embed=self._embed(
                "🟢 System Started",
                f"Monitoring system online at {ts}",
                config.DISCORD_COLOR_GREEN,
            ),
        )
        return True

    def send_system_stop(self) -> bool:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._send(
            CH_ADMIN,
            embed=self._embed(
                "🔴 System Stopped",
                f"Monitoring system shut down at {ts}",
                config.DISCORD_COLOR_RED,
            ),
        )
        return True

    def send_error(self, error_message: str) -> bool:
        self._send(
            CH_ADMIN,
            embed=self._embed(
                "❌ System Error",
                f"```\n{error_message[:1500]}\n```",
                config.DISCORD_COLOR_RED,
            ),
        )
        return True

    def test_connection(self) -> bool:
        if not self._enabled:
            return False
        return self._send(CH_ADMIN, content="🔧 Monitoring system connection test")

    @property
    def is_enabled(self) -> bool:
        return self._enabled
