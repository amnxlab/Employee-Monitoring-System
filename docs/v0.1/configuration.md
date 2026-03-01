# Configuration Reference — v0.1

All configuration lives in `config.py`. Values are overridable via environment variables (loaded from `.env` using `python-dotenv`).

> **Note:** `.env.example` is referenced in the README but has not been committed to the repository as of v0.1. Create a `.env` file manually based on the variables documented below.

---

## Detection

| Constant | Default | Env Var | Description |
|----------|---------|---------|-------------|
| `YOLO_MODEL` | `"yolov8n.pt"` | — | YOLO weights file (auto-downloaded by ultralytics on first run) |
| `PERSON_CONFIDENCE_THRESHOLD` | `0.5` | — | Minimum YOLO confidence to accept a person detection |
| `PERSON_CLASS_ID` | `0` | — | COCO class ID for "person" |
| `YOLO_IMGSZ` | `320` | — | Inference image size (px). Smaller = faster. 320 ≈ 10 ms; 640 ≈ 50 ms on CPU |
| `DETECTION_SKIP_FRAMES` | `5` | — | Run YOLO every N frames; ByteTracker interpolates between |
| `ONNX_THREADS` | `2` | — | ONNX Runtime intra-op threads (prevents CPU saturation) |

---

## Face Recognition

| Constant | Default | Env Var | Description |
|----------|---------|---------|-------------|
| `FACE_SIMILARITY_THRESHOLD` | `0.72` | — | Cosine similarity threshold for a positive match. Higher = stricter (raised from 0.65 to reduce false positives) |
| `FACE_DETECTION_SIZE` | `(640, 640)` | — | InsightFace detection input size |
| `FACE_FRAME_MAX_DIM` | `480` | — | Long side of frame is scaled to this before face detection (performance optimisation) |

---

## Tracking

| Constant | Default | Env Var | Description |
|----------|---------|---------|-------------|
| `TRACK_BUFFER` | `150` | — | Frames to keep a lost track alive without detections (~5 s at 30 fps) |

---

## Binding Persistence

| Constant | Default | Env Var | Description |
|----------|---------|---------|-------------|
| `BINDING_PERSIST_SECONDS` | `7200` | — | After initial face recognition, re-identify purely by body descriptor for this duration (2 hours) |
| `SAME_CAM_HANDOFF_WINDOW` | `300` | — | Same-camera re-ID window in seconds (5 minutes) |
| `SPATIAL_MATCH_PIXELS` | `150` | — | Pixel radius for position-based matching boost during re-ID |

---

## Track Validation

Filters out false detections (hands, objects, etc.) before binding to an employee.

| Constant | Default | Description |
|----------|---------|-------------|
| `MIN_TRACK_AREA` | `3000 px²` | Minimum bounding-box area for a valid person track |
| `TRACK_ASPECT_RATIO_MIN` | `0.2` | Minimum width/height ratio (persons are taller than wide) |
| `TRACK_ASPECT_RATIO_MAX` | `1.0` | Maximum width/height ratio (allows seated/crouching pose) |
| `FACE_IN_UPPER_RATIO` | `0.7` | Face center must be in the upper 70% of the body bounding box |
| `BINDING_IOU_THRESHOLD` | `0.1` | Minimum IoU between face bbox and body bbox to accept a binding |

---

## Attendance State Machine

| Constant | Default | Env Var | Description |
|----------|---------|---------|-------------|
| `TEMP_LOST_TIMEOUT_SECONDS` | `1800` | — | Grace period after losing an employee before automatic clock-out (30 minutes) |
| `DEBOUNCE_FRAMES` | `3` | — | Consecutive visible/invisible frames required before a state transition occurs |

---

## Camera

| Constant | Default | Env Var | Description |
|----------|---------|---------|-------------|
| `CAMERA_INDEX` | `0` | `CAMERA_INDEX` | Single-camera index (used by `register_employee.py` and legacy code) |
| `CAMERA_INDICES` | `[0, 1]` | `CAMERA_INDICES` | Comma-separated list of camera indices for multi-camera mode (e.g. `"0,1,2"`) |
| `FRAME_WIDTH` | `1280` | — | Requested capture width in pixels |
| `FRAME_HEIGHT` | `720` | — | Requested capture height in pixels |
| `CAMERA_FPS` | `30` | — | Requested capture framerate |

---

## Multi-Camera Handoff

| Constant | Default | Description |
|----------|---------|-------------|
| `HANDOFF_WINDOW_SECONDS` | `30` | Seconds after losing a track to attempt histogram handoff on another camera |
| `HISTOGRAM_MATCH_THRESHOLD` | `0.7` | `cv2.compareHist` correlation threshold for cross-camera identity handoff |

---

## Audio Alerts

| Constant | Default | Env Var | Description |
|----------|---------|---------|-------------|
| `AUDIO_ENABLED` | `true` | `AUDIO_ENABLED` | Set to `"false"` to disable all TTS audio |
| `AUDIO_WARNING_DELAY` | `60 s` | — | Seconds before the first spoken warning for an unidentified person |
| `AUDIO_WARNING_INTERVAL` | `300 s` | — | Seconds between repeated warnings |

---

## Discord Webhooks

| Constant | Default | Env Var | Description |
|----------|---------|---------|-------------|
| `DISCORD_WEBHOOK_HERE_GONE` | `""` | `DISCORD_WEBHOOK_HERE_GONE` | Webhook URL for `#here-gone` channel |
| `DISCORD_WEBHOOK_CLOCK_LOGS` | `""` | `DISCORD_WEBHOOK_CLOCK_LOGS` | Webhook URL for `#clock-in-logs` channel |
| `DISCORD_WEBHOOK_ADMIN` | `""` | `DISCORD_WEBHOOK_ADMIN` | Webhook URL for `#admin` channel |
| `DISCORD_WEBHOOK_URL` | `""` | `DISCORD_WEBHOOK_URL` | Legacy single-URL fallback (used if per-channel URLs are not set) |

### Discord Embed Colours (decimal)

| Constant | Value | Usage |
|----------|-------|-------|
| `DISCORD_COLOR_GREEN` | `0x2ECC71` | Clock-in, recovered |
| `DISCORD_COLOR_RED` | `0xE74C3C` | Clock-out, errors |
| `DISCORD_COLOR_ORANGE` | `0xF39C12` | Warnings, temp-lost |
| `DISCORD_COLOR_BLUE` | `0x3498DB` | Info, summaries |
| `DISCORD_COLOR_PURPLE` | `0x9B59B6` | Admin/security |

---

## File Paths

| Constant | Default | Env Var | Description |
|----------|---------|---------|-------------|
| `EMBEDDINGS_PATH` | `data/employees/embeddings.pkl` | `EMBEDDINGS_PATH` | Face embeddings database |
| `LOGS_DIR` | `data/logs/` | `LOGS_DIR` | Attendance JSON log directory |
| `SNAPSHOT_DIR` | `data/snapshots/` | `SNAPSHOT_DIR` | Clock-in snapshot directory |

All three directories are auto-created on startup if they do not exist.

---

## Security

| Constant | Default | Env Var | Description |
|----------|---------|---------|-------------|
| `MANAGER_PASSWORD_HASH` | SHA-256 of default password | `MANAGER_PASSWORD_HASH` | SHA-256 hex digest of the manager password required to quit the application |

To set a custom password:
```python
import hashlib
print(hashlib.sha256("your_password_here".encode()).hexdigest())
```
Paste the output as `MANAGER_PASSWORD_HASH=<hash>` in your `.env`.

---

## Work-Hour Expectations

Used for late-arrival and overtime alerts.

| Constant | Default | Env Var | Description |
|----------|---------|---------|-------------|
| `EXPECTED_START_HOUR` | `9` | `EXPECTED_START_HOUR` | Expected clock-in hour (24-hour, local time) |
| `EXPECTED_END_HOUR` | `17` | `EXPECTED_END_HOUR` | Expected clock-out hour |
| `DAILY_OVERTIME_THRESHOLD` | `28800 s (8 h)` | — | Daily seconds before an overtime alert fires |
| `WEEKLY_OVERTIME_THRESHOLD` | `144000 s (40 h)` | — | Weekly seconds before a weekly overtime alert fires |
| `BREAK_MAX_SECONDS` | `900 s (15 min)` | — | TEMP_LOST durations under this are logged as "break" rather than "absence" |

---

## Sample `.env` File

```dotenv
# Camera
CAMERA_INDEX=0
CAMERA_INDICES=0,1

# Discord (3-channel)
DISCORD_WEBHOOK_HERE_GONE=https://discord.com/api/webhooks/...
DISCORD_WEBHOOK_CLOCK_LOGS=https://discord.com/api/webhooks/...
DISCORD_WEBHOOK_ADMIN=https://discord.com/api/webhooks/...

# Audio
AUDIO_ENABLED=true

# Security (SHA-256 of your chosen password)
MANAGER_PASSWORD_HASH=04773d38e9fd33afcf0da6ddb37875497de06cd6b383d3ec121db2ee4665c7b7

# Work hours
EXPECTED_START_HOUR=9
EXPECTED_END_HOUR=17

# Paths (optional overrides)
# EMBEDDINGS_PATH=data/employees/embeddings.pkl
# LOGS_DIR=data/logs
# SNAPSHOT_DIR=data/snapshots
```
