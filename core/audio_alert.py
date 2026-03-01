"""
Audio alert manager -- multi-scenario, fun, personality-rich warnings.

Scenarios:
  - UNKNOWN_PERSON: New face → ask them to scan for clock-in
  - CLOCK_IN: Greet the employee when they clock in
  - CLOCK_OUT: Farewell when they clock out
  - MISSING: Employee disappeared while clocked in
  - LATE_ARRIVAL: Employee clocked in after expected start hour
  - OVERTIME: Employee has been working 8+ hours
  - RECOVERED: Employee returned from TEMP_LOST
  - BREAK_REMINDER: Been working a long time without a break

All speech runs on a background thread with random voices, rates,
and beep melodies for variety.
"""

import random
import time
import logging
import threading
from typing import Dict, List, Optional

try:
    import pyttsx3
    _PYTTSX3_AVAILABLE = True
except ImportError:
    pyttsx3 = None  # type: ignore
    _PYTTSX3_AVAILABLE = False

import config

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Message pools per scenario
# ═══════════════════════════════════════════════════════════════════════

# {name} is replaced with employee name.

MESSAGES = {
    # ── Friendly prompt: look at camera to clock in ──
    "look_at_camera": [
        "Hey there! Please look at the camera so I can clock you in!",
        "Hi! I see someone new. Could you face the camera for a moment please?",
        "Welcome! Please look towards the camera so I can identify you.",
        "Hello! Face the camera for just a second and I'll get you clocked in!",
        "New person detected! Please glance at the camera so I know who you are.",
        "Hi there! Quick look at the camera and you're all set!",
        "Excuse me! A quick peek at the camera and I'll have you clocked in!",
    ],

    # ── Unknown person (security alert after 5 min) ──
    "unknown_person": [
        "Hey there stranger! I don't know you yet. Come closer so I can see your beautiful face!",
        "Ooh, a mysterious visitor! Please look at the camera so I can figure out who you are!",
        "Beep boop! Unidentified human detected! Please scan your face to clock in!",
        "Who goes there? Friend or foe? Just kidding! Please look at the camera!",
        "New face alert! Come on, don't be shy. Let me see those gorgeous features!",
        "Intruder alert! Just kidding. But seriously, who are you? Face the camera please!",
        "I spy with my little camera eye... someone I don't recognize! Scan your face please!",
        "Welcome, mystery person! Step into the spotlight and show me that face!",
    ],

    # ── Clock-in greetings ──
    "clock_in": [
        "Welcome {name}! Another beautiful day of productivity awaits!",
        "Good to see you, {name}! Let's crush it today!",
        "Hey {name}! The office just got 100 percent more awesome!",
        "{name} has entered the building! Ladies and gentlemen, the star is here!",
        "Top of the morning, {name}! Or afternoon. Or evening. I'm a camera, I don't judge!",
        "Beep boop! {name} detected! Loading enthusiasm... done!",
        "Look who decided to show up! Just kidding {name}, we love you!",
        "{name} is in the house! Let the productivity begin!",
        "Welcome back, {name}! Your desk missed you. It told me so.",
        "Ah, {name}! My favorite human! Don't tell the others I said that.",
    ],

    # ── Clock-in late ──
    "clock_in_late": [
        "Well well well, {name}! Nice of you to finally join us!",
        "Good afternoon, {name}! Oh wait, it's still morning? Could have fooled me!",
        "{name} has arrived! Better late than never, am I right?",
        "Traffic must have been terrible, right {name}? Riiiight?",
        "Alert: {name} has clocked in fashionably late!",
        "{name}! The early bird gets the worm, but you get the leftover coffee!",
        "Oh hey {name}! We were about to send a search party!",
        "Breaking news: {name} has been spotted! Time of arrival: late!",
    ],

    # ── Clock-out farewells ──
    "clock_out": [
        "See you later, {name}! Don't have too much fun without us!",
        "Bye bye {name}! Your desk will keep your seat warm!",
        "{name} is leaving! Everyone wave goodbye! Oh wait, I'm a camera.",
        "Peace out, {name}! Same time tomorrow?",
        "And {name} has left the building! Elvis style!",
        "Goodbye {name}! Sweet dreams of spreadsheets and deadlines!",
        "Later, {name}! May your commute be traffic-free!",
        "{name} out! Mic drop!",
    ],

    # ── Clock-out early ──
    "clock_out_early": [
        "Leaving already, {name}? The fun was just getting started!",
        "{name} bouncing early today! Don't worry, your secret is safe with me!",
        "Whoa {name}, where's the fire? You've still got hours left!",
        "See ya {name}! Sneaking out early, I see! My lips are sealed. Well, speakers.",
    ],

    # ── Missing (disappeared while clocked in) — first time ──
    "missing_first": [
        "Uh oh! {name} has vanished! Did they get abducted by aliens?",
        "Attention! {name} has gone invisible. Are they a ninja now?",
        "Breaking news! {name} is missing in action. Someone check the coffee machine!",
        "Boop boop! Where did {name} go? The cameras miss your beautiful face!",
        "Red alert! {name} has left the building! Or maybe just the frame.",
        "Hey {name}! The cameras can't find you. Come baaack!",
        "Warning! {name} has disappeared. I repeat, {name} has disappeared!",
        "{name} has gone ghost mode! Spooky!",
        "Excuse me, has anyone seen {name}? Asking for a friend... the camera.",
        "Oh no! {name} is gone! Quick, send a search party!",
    ],

    # ── Missing (repeated warnings) ──
    "missing_repeat": [
        "Still can't see {name}. Starting to worry here!",
        "{name} is still missing. This is not a drill! Well, maybe it is.",
        "Hello? {name}? The cameras are lonely without you!",
        "Update: {name} still not found. Have you checked under the desk?",
        "{name}, come back! We promise the cameras won't judge you!",
        "Reminder: {name} is still invisible. Magic trick still going strong!",
        "Yo {name}! The cameras say they miss you. Awkward.",
        "{name} has been gone a while now. Should we send snacks?",
        "Paging {name}! Your desk misses you! Your chair is crying!",
        "Fun fact: {name} has been invisible for a while. Not so fun actually.",
    ],

    # ── Recovered from TEMP_LOST ──
    "recovered": [
        "Welcome back {name}! We thought we lost you there!",
        "{name} has returned from the shadow realm!",
        "Oh there you are {name}! The bathroom line must have been crazy!",
        "{name} is back, baby! Did you miss us?",
        "The prodigal {name} returns! All is forgiven!",
        "{name} back in action! Coffee break over?",
    ],

    # ── Overtime warning ──
    "overtime": [
        "Hey {name}, you've been here over 8 hours! Your couch misses you, go home!",
        "Overtime alert for {name}! Working hard or hardly leaving?",
        "{name}, you're in overtime territory! Your cat is probably plotting revenge!",
        "Whoa {name}, still here? The building is going to start charging you rent!",
        "{name} is still here! Someone tell them it's okay to go home!",
    ],

    # ── Break reminder (3+ hours without a break) ──
    "break_reminder": [
        "Hey {name}, you've been going non-stop! Maybe grab a coffee?",
        "{name}, reminder: humans need breaks too! You're not a robot. Unlike me.",
        "Psst, {name}! Your brain called. It wants a bathroom break.",
        "{name}, take a breather! Even the cameras rest sometimes. Well, they don't. But you should!",
    ],
}

# ── beep patterns ───────────────────────────────────────────────────────
# Fixed beep counts per scenario:
#   1 beep  = leaving / missing / temp-lost
#   2 beeps = clocking out
#   3 beeps = clocking in

_BEEP = (800, 200)  # standard beep tone

_1_BEEP = [_BEEP]
_2_BEEPS = [_BEEP, _BEEP]
_3_BEEPS = [_BEEP, _BEEP, _BEEP]

# Map scenario -> beep pattern
_SCENARIO_MELODIES = {
    "unknown_person": [_1_BEEP],
    "look_at_camera": [_1_BEEP],
    "clock_in":       [_3_BEEPS],
    "clock_in_late":  [_3_BEEPS],
    "clock_out":      [_2_BEEPS],
    "clock_out_early":[_2_BEEPS],
    "missing_first":  [_1_BEEP],
    "missing_repeat": [_1_BEEP],
    "recovered":      [_3_BEEPS],
    "overtime":       [_1_BEEP],
    "break_reminder": [_1_BEEP],
}


def _play_beep_melody(melody):
    """Play a list of (frequency, duration_ms) beeps (Windows only)."""
    try:
        import winsound
        for freq, dur in melody:
            winsound.Beep(freq, dur)
    except Exception:
        pass  # silently skip on non-Windows or errors


# ═══════════════════════════════════════════════════════════════════════
#  Main class
# ═══════════════════════════════════════════════════════════════════════

class AudioAlertManager:
    """
    Multi-scenario, personality-rich audio alert system.

    Call the scenario-specific methods from main.py::

        mgr.play_clock_in(name)
        mgr.play_clock_out(name, is_early=True)
        mgr.play_unknown_person()
        mgr.play_recovered(name)
        mgr.check_and_warn(emp_id, name, is_visible, is_clocked_in)
    """

    def __init__(self):
        self._enabled: bool = getattr(config, "AUDIO_ENABLED", True)
        self._delay: float = getattr(config, "AUDIO_WARNING_DELAY", 60)
        self._interval: float = getattr(config, "AUDIO_WARNING_INTERVAL", 300)

        # Per-employee state for missing warnings
        self._invisible_since: Dict[str, float] = {}
        self._last_warning: Dict[str, float] = {}
        self._warning_count: Dict[str, int] = {}

        # Per-employee state for break reminders
        self._last_break_reminder: Dict[str, float] = {}

        # Overtime tracking
        self._overtime_warned: Dict[str, bool] = {}

        self._engine = None
        self._lock = threading.Lock()
        self._voices = []
        self._voice_ids: List[str] = []

    # ── lifecycle ──────────────────────────────────────────────────────

    def initialize(self) -> bool:
        if not self._enabled:
            logger.info("Audio alerts disabled via config")
            return True

        if not _PYTTSX3_AVAILABLE:
            logger.warning(
                "pyttsx3 not installed -- audio alerts disabled.  "
                "Install with:  pip install pyttsx3"
            )
            self._enabled = False
            return True

        try:
            self._engine = pyttsx3.init()
            self._engine.setProperty("volume", 1.0)
            self._voices = self._engine.getProperty("voices") or []
            self._voice_ids = [v.id for v in self._voices]

            # Prefer female voice (Zira) for a nicer, natural sound
            female_ids = [v.id for v in self._voices
                          if 'zira' in getattr(v, 'name', '').lower()
                          or getattr(v, 'gender', '') == 'Female']
            if female_ids:
                self._voice_ids = female_ids
                self._engine.setProperty('voice', female_ids[0])
                logger.info(f"Selected female voice: {female_ids[0]}")

            voice_info = []
            for v in self._voices:
                name = getattr(v, "name", "unknown")
                voice_info.append(name)

            logger.info(
                f"Audio alerts ready: {len(self._voices)} voice(s) available: "
                f"{', '.join(voice_info[:5])}"
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to initialise pyttsx3: {e} -- audio alerts disabled")
            self._enabled = False
            return True

    def stop(self):
        if self._engine is not None:
            try:
                self._engine.stop()
            except Exception:
                pass
        self._invisible_since.clear()
        self._last_warning.clear()
        self._warning_count.clear()

    # ── scenario triggers ──────────────────────────────────────────────

    def play_unknown_person(self):
        """Trigger when an unidentified person is detected (5-min security alert)."""
        if not self._enabled:
            return
        msg = random.choice(MESSAGES["unknown_person"])
        self._alert_async(msg, "unknown_person")

    def play_look_at_camera(self):
        """Friendly prompt for new person to face the camera for clock-in."""
        if not self._enabled:
            return
        msg = random.choice(MESSAGES["look_at_camera"])
        self._alert_async(msg, "look_at_camera")

    def play_clock_in(self, name: str, is_late: bool = False, employee_id: str = None):
        """Trigger when an employee clocks in."""
        if not self._enabled:
            return
        if is_late:
            msg = random.choice(MESSAGES["clock_in_late"]).format(name=name)
            self._alert_async(msg, "clock_in_late")
        else:
            msg = random.choice(MESSAGES["clock_in"]).format(name=name)
            self._alert_async(msg, "clock_in")
        # Reset tracking state using employee_id (falls back to name)
        key = employee_id or name
        self._overtime_warned.pop(key, None)
        self._last_break_reminder.pop(key, None)

    def play_clock_out(self, name: str, is_early: bool = False):
        """Trigger when an employee clocks out."""
        if not self._enabled:
            return
        if is_early:
            msg = random.choice(MESSAGES["clock_out_early"]).format(name=name)
            self._alert_async(msg, "clock_out_early")
        else:
            msg = random.choice(MESSAGES["clock_out"]).format(name=name)
            self._alert_async(msg, "clock_out")

    def play_recovered(self, name: str):
        """Trigger when an employee returns from TEMP_LOST."""
        if not self._enabled:
            return
        msg = random.choice(MESSAGES["recovered"]).format(name=name)
        self._alert_async(msg, "recovered")

    def play_overtime(self, name: str, employee_id: str = None):
        """Trigger when an employee exceeds 8 hours."""
        if not self._enabled:
            return
        key = employee_id or name
        if self._overtime_warned.get(key):
            return  # only warn once per session
        self._overtime_warned[key] = True
        msg = random.choice(MESSAGES["overtime"]).format(name=name)
        self._alert_async(msg, "overtime")

    def play_break_reminder(self, name: str, employee_id: str = None):
        """Trigger when an employee has been working 3+ hours without a break."""
        if not self._enabled:
            return
        key = employee_id or name
        now = time.time()
        last = self._last_break_reminder.get(key, 0)
        if now - last < 3600:  # at most once per hour
            return
        self._last_break_reminder[key] = now
        msg = random.choice(MESSAGES["break_reminder"]).format(name=name)
        self._alert_async(msg, "break_reminder")

    # ── missing-person checker (called per frame) ──────────────────────

    def check_and_warn(
        self,
        employee_id: str,
        name: str,
        is_visible: bool,
        is_clocked_in: bool,
    ):
        """
        Call once per frame per employee.

        Triggers a fun audio warning if the employee is clocked in but not
        visible on any camera for longer than AUDIO_WARNING_DELAY.
        """
        if not self._enabled:
            return

        now = time.time()

        if is_visible or not is_clocked_in:
            self._invisible_since.pop(employee_id, None)
            self._last_warning.pop(employee_id, None)
            self._warning_count.pop(employee_id, None)
            return

        if employee_id not in self._invisible_since:
            self._invisible_since[employee_id] = now
            return

        elapsed = now - self._invisible_since[employee_id]
        if elapsed < self._delay:
            return

        last = self._last_warning.get(employee_id, 0.0)
        if now - last < self._interval:
            return

        # Fire warning
        self._last_warning[employee_id] = now
        count = self._warning_count.get(employee_id, 0)
        self._warning_count[employee_id] = count + 1

        if count == 0:
            msg = random.choice(MESSAGES["missing_first"]).format(name=name)
            self._alert_async(msg, "missing_first")
        else:
            msg = random.choice(MESSAGES["missing_repeat"]).format(name=name)
            self._alert_async(msg, "missing_repeat")

    # ── internal ───────────────────────────────────────────────────────

    def _start_worker(self):
        """Start the single background TTS worker thread."""
        import queue as _q
        self._alert_queue = _q.Queue(maxsize=3)
        self._worker_running = True
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def _worker_loop(self):
        """Drain the alert queue and play one alert at a time."""
        while self._worker_running:
            try:
                text, scenario = self._alert_queue.get(timeout=0.5)
            except Exception:
                continue
            self._alert(text, scenario)

    def _alert_async(self, text: str, scenario: str = "missing_first"):
        if not hasattr(self, '_alert_queue'):
            self._start_worker()
        try:
            self._alert_queue.put_nowait((text, scenario))
        except Exception:
            pass  # queue full — drop stale alert

    def _alert(self, text: str, scenario: str):
        with self._lock:
            try:
                # 1. Scenario-specific beep melody
                melodies = _SCENARIO_MELODIES.get(scenario, [_1_BEEP])
                _play_beep_melody(random.choice(melodies))

                # 2. Speak with natural female voice
                if self._engine is not None:
                    # Natural speech rate (not too fast, not robotic)
                    rate = random.randint(155, 175)
                    self._engine.setProperty("rate", rate)

                    logger.info(f"Audio [{scenario}]: {text}")
                    self._engine.say(text)
                    self._engine.runAndWait()

                # (no end beep — the scenario beep already played before speech)

            except Exception as e:
                logger.warning(f"TTS error: {e}")
