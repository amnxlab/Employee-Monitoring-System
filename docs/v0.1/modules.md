# Module Reference — v0.1

---

## `main.py` — Entry Point & Orchestrator

### Classes

#### `_CameraPipeline`
Holds the per-camera state for the detection/tracking pipeline.

| Attribute | Type | Description |
|-----------|------|-------------|
| `cam_id` | `int` | Camera index |
| `detector` | `PersonDetector` | Shared reference (not per-camera) |
| `tracker` | `PersonTracker` | Per-camera ByteTrack instance |
| `loop_counter` | `int` | Frame counter for YOLO skip logic |
| `_prev_dets` | `np.ndarray` | Cached detections for skip frames |
| `_prev_scores` | `np.ndarray` | Cached scores for skip frames |

#### `_FaceRecWorker`
Background thread wrapper for face recognition.

| Method | Description |
|--------|-------------|
| `start()` | Launches daemon thread |
| `stop()` | Joins with 2-second timeout |
| `submit(cam_id, frame, tracks)` | Pushes a job; drops previous if queue full (maxsize=1) |
| `latest_faces` | Property — thread-safe copy of last recognised faces |
| `has_pending` | Property — True while a job is in flight |

#### `MonitoringSystem`
Top-level orchestrator. Owns all subsystems.

| Method | Description |
|--------|-------------|
| `initialize()` | Initialises all subsystems in order |
| `run()` | Starts main loop (blocking) |
| `stop()` | Signals shutdown and performs cleanup |
| `_run_loop()` | Per-frame: get frames → detect → track → face-rec → state update → render |
| `_process_camera(cam_id, frame)` | Runs one frame through detect+track for a single camera |
| `_update_attendance(cam_id, tracks)` | Maps tracks → employee IDs → state machine updates |
| `_handle_clock_in(emp_id, ...)` | Records clock-in, saves snapshot, notifies Discord+Audio |
| `_handle_clock_out(emp_id, ...)` | Records clock-out, notifies Discord+Audio |
| `_render_frame(frame, cam_id, ...)` | Draws bounding boxes, labels, status overlay |

---

## `config.py` — Configuration

Reads environment variables from `.env` via `python-dotenv`. All tunables are module-level constants. See [configuration.md](configuration.md) for the full reference.

---

## `core/camera.py` — Camera Interface

### `Camera`
Simple UVC webcam wrapper using OpenCV DirectShow backend (Windows).

| Method | Returns | Description |
|--------|---------|-------------|
| `initialize()` | `bool` | Opens capture, sets resolution + FPS |
| `read()` | `np.ndarray \| None` | Reads one frame |
| `release()` | — | Releases `cv2.VideoCapture` |
| `frame_center` | `(int, int)` | Property — `(width//2, height//2)` |

**Note:** PTZ/movement control has been fully removed. `ptz_controller.py` is a stub/legacy file.

---

## `core/camera_manager.py` — Multi-Camera Manager

### `_CameraThread`
Internal — one background thread per camera; continuously reads frames.

### `CameraManager`
Public interface for multi-camera setups.

| Method | Returns | Description |
|--------|---------|-------------|
| `initialize()` | `bool` | Opens all cameras, starts reader threads |
| `get_latest_frames()` | `Dict[int, ndarray]` | Returns `{cam_id: frame}` for all cameras with frames ready |
| `get_camera_ids()` | `List[int]` | Active camera indices |
| `release()` | — | Stops all threads, releases all cameras |
| `num_cameras` | `int` | Property |

---

## `core/detector.py` — Person Detector

### `Detection` (dataclass)
```python
Detection(bbox=(x1,y1,x2,y2), confidence=0.87, class_id=0)
```
Properties: `center`, `width`, `height`, `area`, `to_xyxy()`, `to_xywh()`

### `PersonDetector`
YOLOv8n person detector. Auto-exports to ONNX on first run for faster CPU inference.

| Method | Returns | Description |
|--------|---------|-------------|
| `initialize()` | `bool` | Loads ONNX model (or exports then loads); falls back to PyTorch |
| `detect(frame)` | `List[Detection]` | Runs inference; filters by `PERSON_CONFIDENCE_THRESHOLD` and class=0 |

**ONNX pipeline:** On first run, exports `yolov8n.pt` → `yolov8n.onnx` at `YOLO_IMGSZ=320`. Subsequent runs load the `.onnx` directly via `onnxruntime` with `ONNX_THREADS=2` to limit CPU saturation.

---

## `core/tracker.py` — Person Tracker

### `Track` (dataclass)
```python
Track(track_id=3, bbox=(x1,y1,x2,y2), confidence=0.9, status=TrackStatus.TRACKED, frames_tracked=42)
```
Properties: `center`, `width`, `height`, `area`

### `TrackStatus` (Enum)
`NEW | TRACKED | LOST`

### `PersonTracker`
ByteTrack implementation via the `supervision` library.

| Method | Returns | Description |
|--------|---------|-------------|
| `initialize()` | `bool` | Creates `sv.ByteTrack` instance |
| `update(detections, scores)` | `List[Track]` | Updates tracker; returns all currently-active tracks |
| `reset()` | — | Clears all track state |

**ByteTrack settings:** `track_activation_threshold=0.25`, `lost_track_buffer=TRACK_BUFFER`, `minimum_matching_threshold=0.8`, `frame_rate=30`

---

## `core/face_recognition.py` — Face Recognition

### `FaceDetection` (dataclass)
```python
FaceDetection(bbox=(x1,y1,x2,y2), embedding=np.ndarray, confidence=0.95, landmarks=None)
```

### `Employee` (dataclass)
```python
Employee(employee_id="EMP001", name="Alice", embedding=np.ndarray)
```

### `EmployeeDatabase`
Manages `embeddings.pkl` persistence.

| Method | Description |
|--------|-------------|
| `add_employee(id, name, embedding)` | Upserts employee; auto-saves |
| `remove_employee(id)` | Deletes by ID; auto-saves |
| `get_employee(id)` | Returns `Employee \| None` |
| `get_all_employees()` | Returns `List[Employee]` |
| `get_name(id)` | Returns name string (or "Unknown") |

### `FaceRecognizer`
Backend: **InsightFace** (`buffalo_l` model) with fallback to **DeepFace**.

| Method | Returns | Description |
|--------|---------|-------------|
| `initialize()` | `bool` | Tries InsightFace; falls back to DeepFace |
| `detect_faces(frame)` | `List[FaceDetection]` | Detects all faces in frame; resizes to `FACE_FRAME_MAX_DIM` first |
| `identify_all(faces)` | `Dict[int, str]` | Maps face index → `employee_id` for all faces above `FACE_SIMILARITY_THRESHOLD` |
| `match_face(embedding)` | `(str, float) \| None` | Best cosine-similarity match from database |

---

## `core/id_binder.py` — Per-Camera ID Binding

### `Binding` (dataclass)
Links a `track_id` to `employee_id` with creation time and validation history.

### `IDBinder`
Per-camera binding with geometry validation.

| Method | Returns | Description |
|--------|---------|-------------|
| `bind(track_id, employee_id)` | — | Creates/updates binding |
| `find_overlapping_track(face_bbox, tracks)` | `int \| None` | Finds the track whose bbox best overlaps the face bbox (IoU ≥ threshold, geometry checks) |
| `get_employee_id(track_id)` | `str \| None` | Lookup by track ID |
| `get_track_id(employee_id)` | `int \| None` | Lookup by employee ID |
| `cleanup_stale(active_track_ids)` | — | Removes bindings for tracks no longer in tracker output |

**Geometry validation filters out non-person objects:**
- `MIN_TRACK_AREA = 3000 px²`
- `TRACK_ASPECT_RATIO_MIN/MAX = 0.2 – 1.0`
- `FACE_IN_UPPER_RATIO = 0.7`: face center must be in upper 70% of body bbox

---

## `core/global_id_binder.py` — Cross-Camera Binding

### `BodyDescriptor` (dataclass)
Lightweight appearance descriptor (~0.2 ms to compute):
- `histogram`: normalised HSV colour histogram (30×32 bins)
- `aspect_ratio`: body width/height
- `avg_upper_color` / `avg_lower_color`: mean BGR of upper/lower body halves
- `position`: last known bbox center
- `timestamp`: last update time

### `GlobalIDBinder`
Cross-camera identity persistence layer.

| Method | Returns | Description |
|--------|---------|-------------|
| `bind(cam_id, track_id, emp_id, frame, bbox)` | — | Creates binding + computes/updates body descriptor |
| `get_employee_id(cam_id, track_id)` | `str \| None` | Direct lookup or descriptor-based re-ID |
| `get_local_binder(cam_id)` | `IDBinder` | Returns camera-specific binder |
| `try_handoff(cam_id, track_id, frame, bbox)` | `str \| None` | Cross-camera histogram handoff |
| `try_same_cam_reid(cam_id, frame, bbox)` | `str \| None` | Same-camera re-ID by descriptor |
| `cleanup(cam_id, active_ids)` | — | Removes stale per-camera bindings |

---

## `core/audio_alert.py` — TTS Audio Alerts

### `AudioAlertManager`
Personality-rich audio alerts with randomised voices, rates, and messages.

**Scenarios:** `look_at_camera`, `unknown_person`, `clock_in`, `clock_in_late`, `clock_out`, `clock_out_early`, `missing`, `overtime`, `recovered`, `break_reminder`

| Method | Description |
|--------|-------------|
| `play(scenario, name=None)` | Queues speech on background thread; de-duplicates per employee |
| `stop()` | Stops TTS engine |

Requires `pyttsx3`. Silently disabled if not installed or `AUDIO_ENABLED=false`.

---

## `core/snapshot.py` — Event Snapshots

| Function | Description |
|----------|-------------|
| `save_snapshot(frame, event_id, bbox=None)` | Saves JPEG with bbox highlight and watermark; returns relative path |
| `cleanup_old_snapshots(max_age_days)` | Deletes snapshots older than N days |

---

## `attendance/state_machine.py` — Attendance FSM

### `AttendanceState` (Enum)
`OUT | DETECTED | CLOCKED_IN | TEMP_LOST | CLOCKED_OUT`

### `AttendanceStateMachine`
Per-employee state machine. See [architecture.md](architecture.md) for state diagram.

| Method | Returns | Description |
|--------|---------|-------------|
| `update(is_visible, current_time)` | `StateTransition \| None` | Advances FSM; returns transition if one occurred |
| `add_transition_callback(fn)` | — | Registers a callback invoked on every transition |
| `clock_in_time` | `float \| None` | Unix timestamp of last clock-in |
| `session_seconds` | `float` | Elapsed session time (updated on clock-out) |

---

## `attendance/timer.py` — Time Tracking

### `EmployeeTime` (dataclass)
Tracks `current_session_start`, `current_session_seconds`, `daily_seconds`, `weekly_seconds` per employee.

### `TimerManager`
Aggregates time for all employees.

| Method | Description |
|--------|-------------|
| `clock_in(employee_id, timestamp)` | Starts session timer |
| `clock_out(employee_id, timestamp, deduct_seconds)` | Ends session; accumulates to daily/weekly |
| `get_session_duration(employee_id)` | Returns current session seconds |
| `get_daily_total(employee_id)` | Returns today's accumulated seconds |
| `get_weekly_total(employee_id)` | Returns this week's accumulated seconds |
| `check_weekly_reset()` | Resets per-employee weekly totals on week rollover |

---

## `attendance/event_logger.py` — JSON Session Logger

### `Session` (dataclass)
One work session from CLOCK_IN to CLOCK_OUT:
```python
Session(
    event_id="EVT-20260228-091532-AB3X",
    employee_id="EMP001",
    clock_in="2026-02-28T09:15:32",
    clock_out="2026-02-28T17:02:11",
    duration_seconds=28359.0,
    interruptions=[Interruption(lost_at=..., recovered_at=...)],
    snapshot_path="data/snapshots/EVT-20260228-091532-AB3X.jpg",
)
```

### `EventLogger`
| Method | Description |
|--------|-------------|
| `log_clock_in(emp_id, emp_name, timestamp, snapshot_path)` | Opens new session, returns `event_id` |
| `log_temp_lost(emp_id, timestamp)` | Adds `Interruption` to open session |
| `log_recovered(emp_id, timestamp)` | Closes current `Interruption` |
| `log_clock_out(emp_id, timestamp, duration)` | Closes session, updates daily/weekly totals, saves JSON |
| `get_weekly_log(emp_id, dt)` | Returns `WeeklyLog` for specified week |
| `generate_weekly_report_embed(...)` | Returns Discord embed dict for weekly summary |

---

## `attendance/logger.py` — Legacy Text Logger

Writes human-readable `.log` files alongside the JSON logs:
```
2026-02-28 09:15:32 | CLOCK_IN
2026-02-28 17:02:11 | CLOCK_OUT | duration: 7h 46m 39s
```
Still active in v0.1; runs in parallel with `EventLogger`.

---

## `integrations/discord_webhook.py` — Discord Notifier

### `DiscordNotifier`
| Method | Channel | Description |
|--------|---------|-------------|
| `notify_clock_in(emp_id, name, time, snapshot)` | `#clock-in-logs` | Rich embed + snapshot attachment |
| `notify_clock_out(emp_id, name, time, duration)` | `#clock-in-logs` | Rich embed with session stats |
| `notify_arrived(name)` | `#here-gone` | Simple presence notification |
| `notify_left(name, duration)` | `#here-gone` | Departure with session time |
| `notify_temp_lost(name)` | `#here-gone` | Tracking signal lost |
| `notify_recovered(name)` | `#here-gone` | Tracking recovered |
| `notify_unknown_person(snapshot)` | `#admin` | Security alert with snapshot |
| `notify_weekly_report(totals, names)` | `#clock-in-logs` | Formatted weekly table |
| `notify_admin(title, description, color)` | `#admin` | Generic admin alert |

All sends are non-blocking (daemon threads). Auto-falls back to single webhook if only `DISCORD_WEBHOOK_URL` is set.

---

## `utils/helpers.py` — Shared Utilities

| Function | Description |
|----------|-------------|
| `calculate_iou(box1, box2)` | Intersection over Union for two xyxy boxes |
| `get_box_center(box)` | Returns `(cx, cy)` |
| `format_duration(seconds)` | `"7h 46m 39s"` |
| `format_hours(seconds)` | `"7.8 hrs"` |
| `get_week_number(dt)` | ISO week number (1–53) |
| `get_week_start_date(dt)` | Monday of the week |
| `is_new_week(last_check, current)` | Returns True if week number changed |

---

## `register_employee.py` — CLI Registration Tool

```bash
python register_employee.py --id EMP001 --name "Alice Smith"  # register
python register_employee.py --list                             # list all
python register_employee.py --remove EMP001                   # delete
python register_employee.py --verify EMP001                   # test match
python register_employee.py --test                            # face detection test
```

Captures 5 face samples; averages embeddings for robustness. Displays live camera feed with face detection overlay during capture.

---

## `watchdog.py` — Auto-Restart Daemon

Monitors `main.py` as a subprocess. Restarts it on crash (non-zero exit). Stops on clean exit (code 0) to respect intentional shutdowns.

| Constant | Value | Description |
|----------|-------|-------------|
| `RESTART_DELAY` | 5 s | Wait before restart |
| `MAX_RAPID_RESTARTS` | 5 | Crash threshold within `RAPID_WINDOW` |
| `RAPID_WINDOW` | 60 s | Window for rapid-restart detection |

On rapid-restart detection → waits 5 minutes before retrying.

Register as Windows Scheduled Task:
```bat
schtasks /create /tn "MonitoringWatchdog" /tr "python <path>/watchdog.py" /sc onlogon /rl highest
```

---

## `check_storage.py` — Storage Inspector

Utility script (undocumented in original README). Inspects stored embeddings and log files. Run directly:
```bash
python check_storage.py
```

---

## `install_face_recognition.py` — Dependency Helper

Tries to install InsightFace; falls back to DeepFace if InsightFace compilation fails. Useful on Windows without C++ Build Tools.

---

## `core/ptz_controller.py` — Legacy PTZ Stub

Originally controlled the OBSBOT Tiny 2 PTZ camera. All movement logic has been removed from `camera.py`. This file may contain unused stubs and is not called from the active codebase.
