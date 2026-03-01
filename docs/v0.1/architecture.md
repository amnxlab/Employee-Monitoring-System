# Architecture — v0.1

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        main.py                              │
│                    MonitoringSystem                         │
│                                                             │
│  CameraManager ──► _CameraPipeline (per cam)               │
│       │                  │ frames                           │
│       │           PersonDetector (shared)                   │
│       │           PersonTracker  (per cam)                  │
│       │                  │ tracks                           │
│       └──────────────────┤                                  │
│                    GlobalIDBinder                           │
│                          │                                  │
│               _FaceRecWorker (bg thread)                    │
│                    FaceRecognizer                           │
│                          │ employee_id                      │
│             AttendanceStateMachine (per employee)           │
│              TimerManager  │  EventLogger                   │
│                            │                                │
│                     DiscordNotifier ──► Discord             │
│                     AudioAlertManager ──► speakers          │
└─────────────────────────────────────────────────────────────┘
```

---

## Threading Model

The system uses several concurrent threads:

| Thread | Class / Origin | Purpose |
|--------|---------------|---------|
| Main thread | `MonitoringSystem._run_loop()` | Per-camera detect + track + render |
| Camera reader threads | `_CameraThread` (per camera) | Grab frames at `CAMERA_FPS`; lock-protected |
| Face recognition thread | `_FaceRecWorker` | Background InsightFace/DeepFace inference |
| Discord sender threads | `DiscordNotifier._send()` | Non-blocking HTTP POST per notification |
| Audio alert thread | `AudioAlertManager` | TTS via pyttsx3, rate/voice randomised |
| Schedule thread | `schedule` library | Weekly report trigger (Sunday 23:55) |

**Synchronisation points:**

- `_CameraThread._lock` — protects `_latest_frame`
- `_FaceRecWorker._q` (maxsize=1) — drops stale frames; at most one pending job
- `_FaceRecWorker._lock` — protects `_latest_faces` / `_latest_cam_id`
- `GlobalIDBinder` internal dicts are accessed only from the main thread and the face-rec thread; `IDBinder` lookups are protected by design (face rec thread calls `bind()`, main thread calls `get_employee_id()`)

---

## Per-Camera Pipeline

Each camera runs a `_CameraPipeline` instance holding:
- A reference to the **shared** `PersonDetector` (one YOLO model for all cameras)
- Its own `PersonTracker` (ByteTrack state is per-camera)
- `loop_counter` for frame-skip logic — YOLO runs every `DETECTION_SKIP_FRAMES` frames; the tracker interpolates in between using previous detections

```
frame N:   detect → track → face_rec_submit → bind → state_machine
frame N+1: (skip detect) → track(prev dets) → bind → state_machine
...
frame N+K: detect → ...
```

---

## Identity Binding Layers

Two binder classes handle different scopes:

### `IDBinder` (per-camera, `core/id_binder.py`)
- Maps ByteTrack `track_id` (integer, per-session) → `employee_id` (string)
- Validates that a detected face actually overlaps with an existing track (IoU check)
- Validates track geometry (minimum area, aspect ratio, face-in-upper-region)
- Cleaned up when tracks are lost

### `GlobalIDBinder` (`core/global_id_binder.py`)
- Wraps one `IDBinder` per camera
- Maintains a `BodyDescriptor` per employee (HSV histogram + upper/lower colour + aspect ratio + last position)
- Enables **2-hour binding persistence**: after initial face recognition, the employee is re-identified by body appearance/position without running FaceRecognizer again
- Implements **cross-camera histogram handoff**: if an employee disappears from one camera and appears on another within `HANDOFF_WINDOW_SECONDS`, colour histogram similarity is used to re-bind without another face scan
- Implements **same-camera re-ID**: within `SAME_CAM_HANDOFF_WINDOW` seconds, a returning track is matched to the last known descriptor of each employee

---

## Attendance State Machine

Per-employee finite state machine in `attendance/state_machine.py`:

```
                     debounce_frames visible
       ┌────────────────────────────────────────┐
       │                                        ▼
  ┌────┴────┐    ┌──────────┐    ┌─────────────────┐
  │   OUT   │───►│ DETECTED │───►│   CLOCKED_IN    │
  └─────────┘    └──────────┘    └────────┬────────┘
       ▲                                  │ debounce_frames invisible
       │                                  ▼
       │                           ┌─────────────┐
       │                           │  TEMP_LOST  │
       │                           └──────┬──────┘
       │        visible again             │ timeout (30 min default)
       │◄─── (back to CLOCKED_IN) ────────┤
       │                                  │
       │                           ┌──────▼──────┐
       │     visible again         │ CLOCKED_OUT │
       └───────────────────────────┤             │
             (new session)         └─────────────┘
```

**State notes:**
- `DETECTED` is a transitional state — it immediately fires the clock-in callback and moves to `CLOCKED_IN` on the **same frame**
- `TEMP_LOST` grace period: `TEMP_LOST_TIMEOUT_SECONDS = 1800` (30 minutes). Re-appearance cancels the countdown without triggering a new session
- A new session starts when the employee is seen again after `CLOCKED_OUT`
- `debounce_frames = 3`: requires 3 consecutive visible/invisible frames before transitioning, preventing flicker

---

## Discord 3-Channel Architecture

```
#here-gone      ← arrivals, departures, temp-lost, recovered
#clock-in-logs  ← formal records, snapshots, daily/weekly reports
#admin          ← security alerts, unknown persons, system errors
```

Environment variables:
```
DISCORD_WEBHOOK_HERE_GONE=...
DISCORD_WEBHOOK_CLOCK_LOGS=...
DISCORD_WEBHOOK_ADMIN=...
# Legacy single-URL fallback:
DISCORD_WEBHOOK_URL=...
```

All sends are fire-and-forget on a daemon thread (non-blocking).

---

## Data Storage

### Face Embeddings
- File: `data/employees/embeddings.pkl`
- Format: Python pickle → `Dict[str, dict]` where each dict holds `employee_id`, `name`, `embedding` (np.ndarray)
- Written by `register_employee.py`; read at startup by `EmployeeDatabase`

### Attendance Logs
- Format: `data/logs/{EMP_ID}_{YEAR}_W{WW}.json`
- Schema: `WeeklyLog` → `List[Session]`
- Each `Session` captures: `event_id`, `clock_in/out` ISO timestamps, `duration_seconds`, optional `snapshot_path`, `List[Interruption]` (TEMP_LOST intervals)
- Written by `EventLogger` (`attendance/event_logger.py`)
- Legacy `.log` text files are written by `AttendanceLogger` (`attendance/logger.py`) — both loggers run simultaneously

### Snapshots
- Directory: `data/snapshots/`
- Format: `{event_id}.jpg` — JPEG with person bounding box and event ID watermark
- Captured on every CLOCK_IN event; path stored in `Session.snapshot_path`
- Auto-cleaned by `cleanup_old_snapshots()` (configurable retention)

### Application Log
- File: `monitoring.log` (rotating, 5 MB × 3 backups)
- Also streams to stdout with UTF-8 encoding

---

## Startup Sequence

1. Load `config.py` (reads `.env`)
2. Initialise `CameraManager` — opens each camera, starts reader threads
3. Initialise `PersonDetector` — loads ONNX model (exports from `.pt` on first run)
4. Initialise per-camera `_CameraPipeline` + `PersonTracker`
5. Initialise `FaceRecognizer` — loads InsightFace or falls back to DeepFace; loads `embeddings.pkl`
6. Initialise `GlobalIDBinder`
7. Start `_FaceRecWorker` background thread
8. Initialise `DiscordNotifier`, `AudioAlertManager`, `EventLogger`, `TimerManager`
9. Send startup notification to Discord `#admin`
10. Register SIGINT/SIGTERM handlers
11. Start `schedule` thread (weekly report)
12. Enter main loop (`_run_loop`)

---

## Shutdown Sequence

1. Signal caught (SIGINT/SIGTERM) or `q` key + password confirmed
2. `_running = False` stops main loop
3. Clock out all active employees; send Discord notifications
4. `_FaceRecWorker.stop()`
5. `CameraManager.release()`
6. Close all OpenCV windows
7. Exit with code `0` (tells watchdog this was a clean shutdown — watchdog will not restart)
