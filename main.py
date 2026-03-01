import sys
import hashlib

try:
    import cv2
except ImportError:
    print("Missing dependency: opencv-python. Install with:")
    print("  python -m pip install opencv-python")
    sys.exit(1)

# ── Limit OpenCV internal threads to prevent CPU thrashing ──
cv2.setNumThreads(2)

import time
import logging
import signal
import threading
import queue
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import schedule

import config
from core import (
    CameraManager,
    PersonDetector,
    FaceRecognizer,
    PersonTracker,
    GlobalIDBinder,
    AudioAlertManager,
)
from core.snapshot import save_snapshot, cleanup_old_snapshots
from attendance import AttendanceStateMachine, AttendanceState, TimerManager, EventLogger
from integrations import DiscordNotifier
from utils.helpers import format_duration, get_week_number

import sys as _sys

from logging.handlers import RotatingFileHandler as _RotatingFileHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(
            open(_sys.stdout.fileno(), mode="w", encoding="utf-8", closefd=False)
        ),
        _RotatingFileHandler(
            "monitoring.log", maxBytes=5*1024*1024, backupCount=3, encoding="utf-8"
        ),
    ]
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Per-camera pipeline state
# ═══════════════════════════════════════════════════════════════════════

class _CameraPipeline:
    """Holds the tracker and frame-skip state for one camera.

    The detector is **shared** across all cameras (saves memory and
    model-load time).  It's passed in from MonitoringSystem.
    """

    def __init__(self, cam_id: int, shared_detector: PersonDetector):
        self.cam_id = cam_id
        self.detector = shared_detector   # shared, NOT per-camera
        self.tracker = PersonTracker()
        self.loop_counter = 0
        self._prev_dets = np.empty((0, 4))
        self._prev_scores = np.empty(0)

    def initialize(self) -> bool:
        # Detector is initialized once in MonitoringSystem
        if not self.tracker.initialize():
            logger.error(f"Tracker init failed for camera {self.cam_id}")
            return False
        return True


# ═══════════════════════════════════════════════════════════════════════
#  Background face-recognition worker (shared across cameras)
# ═══════════════════════════════════════════════════════════════════════

class _FaceRecWorker:
    """
    Runs face recognition in a background thread.

    Accepts (cam_id, frame, tracks) and processes only unbound tracks.
    """

    def __init__(self, face_recognizer: FaceRecognizer, global_binder: GlobalIDBinder):
        self._recognizer = face_recognizer
        self._binder = global_binder
        self._q: queue.Queue = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._latest_faces: list = []
        self._latest_cam_id: Optional[int] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._pending_request = False

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def submit(self, cam_id: int, frame: np.ndarray, tracks: list):
        """Submit a frame from a specific camera for face recognition."""
        try:
            self._q.get_nowait()
        except queue.Empty:
            pass
        self._q.put((cam_id, frame, tracks))
        self._pending_request = True

    @property
    def latest_faces(self):
        with self._lock:
            return list(self._latest_faces)

    @property
    def latest_cam_id(self):
        with self._lock:
            return self._latest_cam_id

    @property
    def has_pending(self):
        return self._pending_request

    def _loop(self):
        while self._running:
            try:
                item = self._q.get(timeout=0.1)
            except queue.Empty:
                continue

            cam_id, frame, all_tracks = item
            try:
                faces = self._recognizer.detect_faces(frame)
                matches = self._recognizer.identify_all(faces)

                # Build set of employee IDs already actively bound on ANY
                # camera so we never re-bind them (prevents identity swaps
                # when two people are near each other).
                already_bound = set(self._binder.get_visible_employees())

                # Also track which track_ids we assign in THIS batch to
                # prevent two faces from claiming the same track.
                claimed_tracks: set = set()

                # Bind matched faces through the global binder
                for face_idx, employee_id in matches.items():
                    # Skip employees already bound to an active track
                    if employee_id in already_bound:
                        logger.debug(
                            f"[FACE-SKIP] {employee_id} already bound, "
                            f"skipping re-bind on cam {cam_id}"
                        )
                        continue

                    face = faces[face_idx]
                    # Find overlapping track using local binder logic
                    local = self._binder.get_local_binder(cam_id)
                    track_id = local.find_overlapping_track(face.bbox, all_tracks)
                    if track_id is not None:
                        # Prevent two different employees from claiming
                        # the same track in one recognition pass
                        if track_id in claimed_tracks:
                            logger.debug(
                                f"[FACE-SKIP] track {track_id} already "
                                f"claimed this pass, skipping {employee_id}"
                            )
                            continue
                        claimed_tracks.add(track_id)

                        # Find the Track object for bbox
                        track_obj = next(
                            (t for t in all_tracks if t.track_id == track_id), None
                        )
                        bbox = track_obj.bbox if track_obj else None
                        self._binder.bind(
                            cam_id, track_id, employee_id, frame, bbox
                        )
                        logger.info(
                            f"[FACE-BIND] {employee_id} -> cam {cam_id} "
                            f"track {track_id}"
                        )

                with self._lock:
                    self._latest_faces = faces
                    self._latest_cam_id = cam_id
                self._pending_request = False

                if matches:
                    logger.info(
                        f"Face recognition: {len(matches)} match(es) "
                        f"from {len(faces)} face(s) on cam {cam_id}"
                    )
            except Exception as e:
                logger.debug(f"BG face rec error: {e}")
                self._pending_request = False


# ═══════════════════════════════════════════════════════════════════════
#  Main monitoring system
# ═══════════════════════════════════════════════════════════════════════

class MonitoringSystem:
    """
    Multi-camera monitoring system.

    Handles:
    - Multi-camera video capture (threaded)
    - Per-camera person detection and tracking
    - Cross-camera person handoff via histogram matching
    - Face recognition and global ID binding
    - Attendance state management
    - Audio warnings (TTS) for missing employees
    - Discord notifications
    - Weekly report scheduling
    """

    def __init__(self):
        self.camera_manager = CameraManager()
        self.face_recognizer = FaceRecognizer()
        self.global_binder = GlobalIDBinder()
        self.audio_alert = AudioAlertManager()

        self.pipelines: Dict[int, _CameraPipeline] = {}

        self.timer_manager = TimerManager()
        self.event_logger = EventLogger()
        self.discord = DiscordNotifier()

        self._face_worker: Optional[_FaceRecWorker] = None

        self.state_machines: Dict[str, AttendanceStateMachine] = {}

        self.running = False
        self.debug_display = True
        self._last_week = get_week_number()
        self._last_day = datetime.now().strftime("%Y-%m-%d")
        # Frame counter for periodic histogram updates (every 60 frames ~ 2s)
        self._hist_update_interval = 60
        # Latest frame per camera (for snapshots on clock-in)
        self._latest_frames: Dict[int, np.ndarray] = {}
        # Unknown person tracker: {(cam_id, track_id): {first_seen, first_snap, alerted}}
        self._unknown_tracker: Dict[tuple, dict] = {}
        self._unknown_grace_seconds = getattr(config, 'UNKNOWN_GRACE_SECONDS', 300)  # 5 min
        # Round-robin counter: which camera submits to face-rec this iteration
        self._face_rec_cam_turn: int = 0
        # Exit code: 0 = clean password-confirmed quit, 1 = crash/kill
        self._exit_code = 1

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    # ── init ───────────────────────────────────────────────────────────

    def _signal_handler(self, signum, frame):
        logger.info("Shutdown signal received")
        self.stop()

    def initialize(self) -> bool:
        """Initialize all components."""
        logger.info("Initializing multi-camera monitoring system...")

        # Cameras
        if not self.camera_manager.initialize():
            logger.error("Failed to initialize cameras")
            return False

        # Shared detector (ONE model for all cameras)
        self._shared_detector = PersonDetector()
        if not self._shared_detector.initialize():
            logger.error("Failed to initialize shared person detector")
            return False

        # Per-camera pipelines (trackers only, detector is shared)
        for cam_id in self.camera_manager.get_camera_ids():
            pipe = _CameraPipeline(cam_id, self._shared_detector)
            if not pipe.initialize():
                return False
            self.pipelines[cam_id] = pipe
            self.global_binder.register_camera(cam_id)
            logger.info(f"Pipeline ready for camera {cam_id}")

        # Face recognizer (shared)
        if not self.face_recognizer.initialize():
            logger.error("Failed to initialize face recognizer")
            return False

        # Face-rec background worker
        self._face_worker = _FaceRecWorker(self.face_recognizer, self.global_binder)
        self._face_worker.start()

        # Audio alerts
        self.audio_alert.initialize()

        # State machines
        self._init_state_machines()
        self._schedule_reports()

        logger.info(
            f"Multi-camera monitoring system ready – "
            f"{self.camera_manager.num_cameras} camera(s)"
        )
        return True

    # ── state machines ─────────────────────────────────────────────────

    def _init_state_machines(self):
        employee_ids = self.face_recognizer.get_all_employee_ids()
        for emp_id in employee_ids:
            if emp_id not in self.state_machines:
                sm = AttendanceStateMachine(emp_id)
                sm.add_transition_callback(self._on_state_transition)
                self.state_machines[emp_id] = sm
        logger.info(f"Initialized state machines for {len(self.state_machines)} employees")

    def _on_state_transition(self, transition):
        emp_id = transition.employee_id
        name = self.face_recognizer.get_employee_name(emp_id)
        timestamp = datetime.fromtimestamp(transition.timestamp)

        if transition.to_state == AttendanceState.CLOCKED_IN:
            if transition.from_state in [AttendanceState.DETECTED, AttendanceState.OUT]:
                # New session: clock-in + snapshot
                self.timer_manager.clock_in(emp_id, transition.timestamp)
                snap_path = self._take_clock_in_snapshot(emp_id)
                event_id = self.event_logger.log_clock_in(
                    emp_id, name, snapshot_path=snap_path, timestamp=timestamp
                )
                is_late = timestamp.hour >= getattr(config, 'EXPECTED_START_HOUR', 9) + 1
                self.discord.send_clock_in(
                    name, timestamp, snapshot_path=snap_path, is_late=is_late
                )
                logger.info(f"{name} clocked in [{event_id}]")
                # Audio: clock-in greeting
                self.audio_alert.play_clock_in(name, is_late=is_late)
            elif transition.from_state == AttendanceState.TEMP_LOST:
                # Same session: recovered (no new clock-in)
                parent = self.event_logger.get_active_event_id(emp_id)
                self.event_logger.log_recovered(emp_id, parent, timestamp)
                self.discord.send_recovered(name)
                logger.info(f"{name} recovered from temp lost")
                # Audio: welcome back
                self.audio_alert.play_recovered(name)

        elif transition.to_state == AttendanceState.TEMP_LOST:
            parent = self.event_logger.get_active_event_id(emp_id)
            self.event_logger.log_temp_lost(emp_id, parent, timestamp)
            self.discord.send_temp_lost(name)
            logger.info(f"{name} temporarily lost")

        elif transition.to_state == AttendanceState.CLOCKED_OUT:
            duration = self.timer_manager.get_current_session(emp_id, transition.timestamp)
            deduct = config.TEMP_LOST_TIMEOUT_SECONDS if transition.from_state == AttendanceState.TEMP_LOST else 0
            self.timer_manager.clock_out(emp_id, transition.timestamp, deduct)
            final_duration = self.timer_manager.get_last_session_duration(emp_id)
            parent = self.event_logger.get_active_event_id(emp_id)
            self.event_logger.log_clock_out(
                emp_id, parent, duration_seconds=final_duration, timestamp=timestamp
            )
            is_early = timestamp.hour < getattr(config, 'EXPECTED_END_HOUR', 17)
            self.discord.send_clock_out(
                name, final_duration, timestamp, is_early=is_early
            )
            logger.info(f"{name} clocked out - session: {format_duration(final_duration)}")
            # Audio: farewell
            self.audio_alert.play_clock_out(name, is_early=is_early)

    def _take_clock_in_snapshot(self, emp_id: str) -> Optional[str]:
        """Capture a snapshot from the camera where the employee was detected."""
        # Find which camera the employee is bound to
        binding = self.global_binder._employee_bindings.get(emp_id)
        if binding is None:
            return None
        frame = self._latest_frames.get(binding.cam_id)
        if frame is None:
            return None
        # Generate a temporary event ID for filename (will be replaced by real one)
        from attendance.event_logger import _generate_event_id
        event_id = _generate_event_id()
        return save_snapshot(frame, event_id, person_bbox=None)

    # ── reports and daily summary ──────────────────────────────────────

    def _schedule_reports(self):
        schedule.every().sunday.at("23:59").do(self._send_weekly_report)
        schedule.every().day.at("23:55").do(self._send_daily_summary)
        logger.info("Reports scheduled: daily 23:55, weekly Sunday 23:59")

    def _send_daily_summary(self):
        """Log and notify daily totals for all employees."""
        today = datetime.now().strftime("%Y-%m-%d")
        employee_ids = self.face_recognizer.get_all_employee_ids()
        lines = [f"Daily Summary for {today}", "-" * 30]
        for emp_id in employee_ids:
            name = self.face_recognizer.get_employee_name(emp_id)
            daily_secs = self.timer_manager.get_daily_total(emp_id)
            self.event_logger.log_daily_summary(emp_id, name, daily_secs, today)
            hours = daily_secs / 3600
            lines.append(f"  {name}: {hours:.1f} hrs")
            # Overtime check
            if daily_secs > config.DAILY_OVERTIME_THRESHOLD:
                overtime = (daily_secs - config.DAILY_OVERTIME_THRESHOLD) / 3600
                lines.append(f"    ** OVERTIME: +{overtime:.1f} hrs **")
        report = "\n".join(lines)
        self.discord.send_message(report) if hasattr(self.discord, 'send_message') else None
        logger.info(f"Daily summary sent for {today}")

        # Auto-prune snapshots older than 30 days
        cleanup_old_snapshots(max_age_days=30)

    def _send_weekly_report(self):
        weekly_totals = self.timer_manager.get_all_weekly_totals()
        employee_names = {
            emp_id: self.face_recognizer.get_employee_name(emp_id)
            for emp_id in weekly_totals.keys()
        }
        report = self.event_logger.generate_weekly_report(
            list(weekly_totals.keys()), employee_names
        )
        self.discord.send_weekly_summary(weekly_totals, employee_names)
        logger.info("Weekly report sent")

    def _check_weekly_reset(self):
        current_week = get_week_number()
        if current_week != self._last_week:
            logger.info(f"New week detected: {self._last_week} -> {current_week}")
            self._send_weekly_report()
            self.timer_manager.reset_weekly()
            self._last_week = current_week

    # ── main loop ──────────────────────────────────────────────────────

    def run(self):
        """Main processing loop – processes all cameras each iteration."""
        self.running = True
        self.discord.send_system_start()
        logger.info("Starting multi-camera main loop...")

        frame_count = 0
        fps_start_time = time.time()
        fps = 0.0
        global_loop = 0
        # Cap main loop at 30 iterations/sec – YOLO only runs every
        # DETECTION_SKIP_FRAMES anyway, so faster looping wastes CPU.
        target_loop_fps = 30  # user-requested minimum
        target_loop_dt = 1.0 / target_loop_fps

        try:
            while self.running:
                loop_start = time.perf_counter()
                schedule.run_pending()
                self._check_weekly_reset()

                current_time = time.time()
                frames = self.camera_manager.get_latest_frames()

                if not frames:
                    time.sleep(0.01)
                    continue

                # Store half-res copies for snapshot access (saves ~75% memory)
                for _cid, _fr in frames.items():
                    _h, _w = _fr.shape[:2]
                    self._latest_frames[_cid] = cv2.resize(_fr, (_w // 2, _h // 2))

                # Accumulate per-camera display data for tiled view
                display_data: Dict[int, dict] = {}

                for cam_id, frame in frames.items():
                    pipe = self.pipelines.get(cam_id)
                    if pipe is None:
                        continue

                    # ── detection (with per-camera frame skipping) ──
                    # Each camera independently skips DETECTION_SKIP_FRAMES.
                    # Removed per-camera alternation: it halved effective
                    # detection rate per camera and caused track fragmentation.
                    skip = getattr(config, "DETECTION_SKIP_FRAMES", 1)
                    should_detect = (pipe.loop_counter % skip == 0)

                    if should_detect:
                        detections, scores = pipe.detector.detect_with_scores(frame)
                    else:
                        detections, scores = pipe._prev_dets, pipe._prev_scores
                    pipe._prev_dets, pipe._prev_scores = detections, scores
                    pipe.loop_counter += 1

                    # ── tracking ──
                    tracks = pipe.tracker.update(detections, scores)
                    active_ids = pipe.tracker.get_active_track_ids()

                    # ── cleanup lost bindings on this camera ──
                    self.global_binder.cleanup_lost_tracks(cam_id, active_ids)

                    # ── descriptor-based re-ID (same-cam + cross-cam) ──
                    unbound = self.global_binder.get_unbound_tracks(cam_id, tracks)
                    for track in unbound:
                        self.global_binder.attempt_handoff(cam_id, track, frame)

                    # ── face rec for UNBOUND tracks only ──
                    # Face recognition only fires when there are unbound
                    # tracks.  Already-bound tracks are maintained by the
                    # body-descriptor system, not by re-running face rec
                    # (which caused identity swaps between nearby people).
                    # Round-robin: only ONE camera submits per iteration
                    # to prevent the queue-drop overwrite bug.
                    cam_ids = list(frames.keys())
                    still_unbound = self.global_binder.get_unbound_tracks(cam_id, tracks)
                    is_my_turn = (cam_ids.index(cam_id) == self._face_rec_cam_turn % len(cam_ids))
                    if (bool(still_unbound)
                            and is_my_turn
                            and not self._face_worker.has_pending):
                        # Pass original frame — detect_faces() handles its
                        # own downscale + upscale so face bboxes are returned
                        # in original-frame coordinates, matching track bboxes.
                        self._face_worker.submit(cam_id, frame, tracks)

                    # ── periodically refresh histograms for bound tracks ──
                    if global_loop % self._hist_update_interval == 0:
                        for track in tracks:
                            emp = self.global_binder.get_employee_for_track(cam_id, track.track_id)
                            if emp:
                                self.global_binder.update_histogram(
                                    cam_id, track.track_id, frame, track.bbox
                                )

                    # Collect faces for this camera's debug overlay
                    cam_faces = []
                    if self._face_worker.latest_cam_id == cam_id:
                        cam_faces = self._face_worker.latest_faces

                    display_data[cam_id] = {
                        "frame": frame,
                        "tracks": tracks,
                        "faces": cam_faces,
                    }

                # Advance face-rec round-robin so next iteration a different camera submits
                self._face_rec_cam_turn += 1

                # ── global attendance state update ──
                for emp_id, sm in self.state_machines.items():
                    is_visible = self.global_binder.is_employee_visible_any_camera(emp_id)
                    sm.update(is_visible, current_time)

                    # Audio: missing-person warnings
                    name = self.face_recognizer.get_employee_name(emp_id)
                    self.audio_alert.check_and_warn(
                        emp_id, name, is_visible, sm.is_working
                    )

                    # Audio: overtime and break reminders
                    if sm.is_working and is_visible:
                        session_secs = self.timer_manager.get_current_session(
                            emp_id, current_time
                        )
                        if session_secs > getattr(config, 'DAILY_OVERTIME_THRESHOLD', 28800):
                            self.audio_alert.play_overtime(name)
                            self.discord.send_overtime_alert(name, session_secs / 3600)
                        elif session_secs > 3 * 3600:  # 3 hours nonstop
                            self.audio_alert.play_break_reminder(name)

                # Unknown person: 5-minute grace period with first+last snapshot
                # Cross-camera dedup: if the same physical person is visible on
                # two cameras, we treat it as ONE unknown person, not two.
                import tempfile, os
                from core.global_id_binder import compute_body_descriptor, compare_descriptors as _cmp_desc
                active_unknown_keys = set()
                for cam_id in self.pipelines:
                    cam_tracks = display_data.get(cam_id, {}).get('tracks', [])
                    unbound = self.global_binder.get_unbound_tracks(cam_id, cam_tracks)

                    for track in unbound:
                        key = (cam_id, track.track_id)
                        active_unknown_keys.add(key)

                        if key not in self._unknown_tracker:
                            # Cross-camera dedup: check if an existing unknown
                            # entry on another camera matches this person's
                            # body descriptor.  If so, reuse that entry.
                            _frame_for_desc = frames.get(cam_id)
                            _matched_existing = None
                            if _frame_for_desc is not None and self._unknown_tracker:
                                _new_desc = compute_body_descriptor(_frame_for_desc, track.bbox)
                                if _new_desc.histogram is not None:
                                    for _ekey, _eentry in self._unknown_tracker.items():
                                        _edesc = _eentry.get('descriptor')
                                        if _edesc is not None and _edesc.histogram is not None:
                                            _sim = _cmp_desc(_new_desc, _edesc, is_cross_camera=True)
                                            if _sim > 0.55:
                                                _matched_existing = _ekey
                                                break
                                    # Store the descriptor for future dedup checks
                                    _stored_desc = _new_desc
                                else:
                                    _stored_desc = None
                            else:
                                _stored_desc = None

                            if _matched_existing is not None:
                                # Same person already tracked on another cam — link this key
                                # to the existing entry so it shares the same first_seen / alert.
                                self._unknown_tracker[key] = self._unknown_tracker[_matched_existing]
                                active_unknown_keys.add(_matched_existing)
                                logger.debug(
                                    f"Unknown dedup: cam {cam_id} track {track.track_id} "
                                    f"matches existing {_matched_existing}"
                                )
                            else:
                                # Genuinely new unknown person
                                self.audio_alert.play_look_at_camera()

                                # Capture first snapshot for potential security alert later
                                first_snap = None
                                _frame = self._latest_frames.get(cam_id)
                                if _frame is not None:
                                    first_snap = os.path.join(
                                        tempfile.gettempdir(),
                                        f"unknown_first_cam{cam_id}_t{track.track_id}_{int(current_time)}.jpg"
                                    )
                                    cv2.imwrite(first_snap, _frame)
                                self._unknown_tracker[key] = {
                                    'first_seen': current_time,
                                    'first_snap': first_snap,
                                    'alerted': False,
                                    'descriptor': _stored_desc,
                                }
                                logger.info(f"Unknown person tracked: cam {cam_id}, track {track.track_id}")

                        else:
                            entry = self._unknown_tracker[key]
                            elapsed = current_time - entry['first_seen']
                            if elapsed >= self._unknown_grace_seconds and not entry['alerted']:
                                # 5 min passed — capture last snapshot and send alert
                                last_snap = None
                                _frame = self._latest_frames.get(cam_id)
                                if _frame is not None:
                                    last_snap = os.path.join(
                                        tempfile.gettempdir(),
                                        f"unknown_last_cam{cam_id}_t{track.track_id}_{int(current_time)}.jpg"
                                    )
                                    cv2.imwrite(last_snap, _frame)

                                self.audio_alert.play_unknown_person()
                                self.discord.send_unknown_person(
                                    cam_id,
                                    first_snapshot=entry['first_snap'],
                                    last_snapshot=last_snap,
                                    duration_minutes=elapsed / 60,
                                )
                                entry['alerted'] = True
                                logger.warning(
                                    f"Unknown person alert sent: cam {cam_id}, "
                                    f"track {track.track_id}, present for {elapsed/60:.1f} min"
                                )

                                # Cleanup temp files
                                for snap in [entry['first_snap'], last_snap]:
                                    if snap and os.path.exists(snap):
                                        try:
                                            os.remove(snap)
                                        except OSError:
                                            pass

                # Cleanup tracker: remove tracks that disappeared or got bound
                stale = [k for k in self._unknown_tracker if k not in active_unknown_keys]
                for k in stale:
                    entry = self._unknown_tracker.pop(k)
                    # Delete first snapshot if never alerted
                    if not entry['alerted'] and entry.get('first_snap'):
                        try:
                            os.remove(entry['first_snap'])
                        except OSError:
                            pass

                # ── FPS ──
                frame_count += 1
                global_loop += 1
                if frame_count % 30 == 0:
                    elapsed = time.time() - fps_start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    frame_count = 0
                    fps_start_time = time.time()

                # ── debug display ──
                if self.debug_display and display_data:
                    tiled = self._build_tiled_display(display_data, fps)
                    cv2.imshow("Monitoring System", tiled)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    if self._verify_password():
                        logger.info("Password-confirmed quit")
                        self._exit_code = 0  # clean exit for watchdog
                        break
                    else:
                        logger.info("Quit rejected: wrong password")
                elif key == ord('d'):
                    self.debug_display = not self.debug_display
                    logger.info(f"Debug display: {self.debug_display}")

                # ── check for day rollover (reuse current_time from above) ──
                today = datetime.fromtimestamp(current_time).strftime("%Y-%m-%d")
                if today != self._last_day:
                    self.timer_manager.reset_daily()
                    self._last_day = today

                # ── throttle to target FPS to save CPU ──
                loop_elapsed = time.perf_counter() - loop_start
                sleep_remain = target_loop_dt - loop_elapsed
                if sleep_remain > 0:
                    time.sleep(sleep_remain)

        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
            self.discord.send_error(str(e))
        finally:
            self.stop()

    # ── tiled debug view ───────────────────────────────────────────────

    def _build_tiled_display(
        self, display_data: Dict[int, dict], fps: float
    ) -> np.ndarray:
        """Build a side-by-side view of all cameras with overlays."""
        panels: List[np.ndarray] = []
        target_h = 360  # each panel height for the tiled view

        for cam_id in sorted(display_data.keys()):
            data = display_data[cam_id]
            panel = self._draw_cam_overlay(
                cam_id, data["frame"], data["tracks"], data["faces"], fps
            )
            # Resize to uniform height
            h, w = panel.shape[:2]
            scale = target_h / h
            panel = cv2.resize(panel, (int(w * scale), target_h))
            panels.append(panel)

        if not panels:
            return np.zeros((target_h, 640, 3), dtype=np.uint8)

        # Side-by-side
        tiled = np.hstack(panels)

        # Draw global status bar at bottom
        bar_h = 30 + 25 * len(self.state_machines)
        bar = np.zeros((bar_h, tiled.shape[1], 3), dtype=np.uint8)
        y_off = 20
        for emp_id, sm in self.state_machines.items():
            name = self.face_recognizer.get_employee_name(emp_id)
            state = sm.current_state.value
            if sm.current_state == AttendanceState.CLOCKED_IN:
                duration = self.timer_manager.get_current_session(emp_id)
                status = f"{name}: {state} ({format_duration(duration)})"
                color = (0, 255, 0)
            elif sm.current_state == AttendanceState.TEMP_LOST:
                remaining = sm.get_temp_lost_remaining()
                status = f"{name}: {state} ({remaining:.0f}s remaining)"
                color = (0, 165, 255)
            else:
                status = f"{name}: {state}"
                color = (128, 128, 128)
            cv2.putText(bar, status, (10, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            y_off += 25

        return np.vstack([tiled, bar])

    def _draw_cam_overlay(
        self,
        cam_id: int,
        frame: np.ndarray,
        tracks: list,
        faces: list,
        fps: float,
    ) -> np.ndarray:
        """Draw bounding boxes and labels on a single camera's frame."""
        display = frame.copy()

        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            emp_id = self.global_binder.get_employee_for_track(cam_id, track.track_id)
            if emp_id:
                color = (0, 255, 0)
                name = self.face_recognizer.get_employee_name(emp_id)
                label = f"{name} ({track.track_id})"
            else:
                color = (0, 165, 255)
                label = f"Person {track.track_id}"
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        for face in faces:
            x1, y1, x2, y2 = face.bbox
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 0), 1)

        # Camera label + FPS
        cv2.putText(display, f"Cam {cam_id} | FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return display

    # ── shutdown ───────────────────────────────────────────────────────

    # ── password verification ───────────────────────────────────────────

    def _verify_password(self) -> bool:
        """Show an OpenCV password overlay and verify against manager hash."""
        pwd_buffer = ""
        prompt_active = True
        result = False

        while prompt_active:
            # Draw password overlay on a dark frame
            overlay = np.zeros((200, 500, 3), dtype=np.uint8)
            cv2.putText(overlay, "Enter manager password:", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            masked = "*" * len(pwd_buffer)
            cv2.putText(overlay, masked, (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(overlay, "[Enter] confirm  |  [Esc] cancel", (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            cv2.imshow("Monitoring System", overlay)

            key = cv2.waitKey(0) & 0xFF
            if key == 13:  # Enter
                pwd_hash = hashlib.sha256(pwd_buffer.encode()).hexdigest()
                result = pwd_hash == config.MANAGER_PASSWORD_HASH
                if not result:
                    # Wrong password -> alert admin
                    self.discord.send_wrong_password()
                    logger.warning("Failed shutdown attempt")
                prompt_active = False
            elif key == 27:  # Esc to cancel
                prompt_active = False
                result = False
            elif key == 8:  # Backspace
                pwd_buffer = pwd_buffer[:-1]
            elif 32 <= key <= 126:  # Printable ASCII
                pwd_buffer += chr(key)

        return result

    # ── shutdown ───────────────────────────────────────────────────────

    def stop(self):
        if not self.running:
            return
        self.running = False
        logger.info("Stopping monitoring system...")

        if self._face_worker:
            self._face_worker.stop()

        self.audio_alert.stop()

        for emp_id, sm in self.state_machines.items():
            if sm.is_working:
                sm.force_clock_out()

        self.discord.send_system_stop()
        self.camera_manager.release()
        cv2.destroyAllWindows()
        logger.info("Monitoring system stopped")


# ═══════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════

def main():
    """Entry point."""
    system = MonitoringSystem()

    if not system.initialize():
        logger.error("Failed to initialize system")
        print("\nTo fix face recognition, install a backend (no Build Tools needed):")
        print("  python -m pip install deepface")
        print("Or run:  python install_face_recognition.py")
        print("\nMake sure pyttsx3 is installed for audio alerts:")
        print("  python -m pip install pyttsx3")
        sys.exit(1)

    system.run()
    # Exit with the appropriate code so the watchdog knows what to do:
    #   0 = clean password-confirmed quit (watchdog stops)
    #   1 = crash/kill (watchdog restarts)
    sys.exit(system._exit_code)


if __name__ == "__main__":
    main()
