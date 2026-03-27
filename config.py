import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent

# Detection
YOLO_MODEL = "yolov8n.pt"
PERSON_CONFIDENCE_THRESHOLD = 0.5
PERSON_CLASS_ID = 0
YOLO_IMGSZ = 320  # smaller = faster; 320 ~10ms vs 640 ~50ms on CPU
DETECTION_SKIP_FRAMES = 3  # run YOLO every N frames; lowered from 5 for denser lab environments
ONNX_THREADS = 2  # threads for ONNX Runtime inference (limits CPU saturation)

# Face Recognition
FACE_SIMILARITY_THRESHOLD = 0.78  # Raised from 0.72 to reduce false positives in crowded lab
FACE_DETECTION_SIZE = (640, 640)
FACE_FRAME_MAX_DIM = 480  # resize long side before DeepFace to save time
FACE_MIN_SIZE = 40          # minimum face bbox dimension (px); smaller faces are too blurry/distant
FACE_CONFIRM_PASSES = 2     # require same employee matched on N independent passes before binding

# Tracking
TRACK_BUFFER = 450  # frames to keep lost tracks alive (~15s at 30fps); raised for desk-work occlusions

# Persistent body tracking -- gap-filler only, NOT a substitute for face recognition
# Keep these values short: body colour matching causes identity swaps when people stand close.
# ByteTrack handles continuous following; body descriptors only bridge momentary occlusions.
BINDING_PERSIST_SECONDS = 60      # how long an employee stays in _remembered; >= HANDOFF_WINDOW_SECONDS
SAME_CAM_HANDOFF_WINDOW = 15      # same-camera body re-ID window (kept short to prevent close-person swaps)
HANDOFF_WINDOW_SECONDS = 60       # cross-camera body re-ID window (person walks between cameras)
SPATIAL_MATCH_PIXELS = 150        # px radius for position-based matching boost

# Track Validation (to prevent binding hands/objects as people)
MIN_TRACK_AREA = 3000  # minimum pixels^2 for a valid person track (lowered for distance)
TRACK_ASPECT_RATIO_MIN = 0.2  # width/height min (person is taller than wide)
TRACK_ASPECT_RATIO_MAX = 1.0  # width/height max (allow sitting/crouching)
FACE_IN_UPPER_RATIO = 0.7  # face center must be in upper 70% of track bbox
BINDING_IOU_THRESHOLD = 0.25  # raised from 0.1 to require meaningful face-track overlap

# State Machine
TEMP_LOST_TIMEOUT_SECONDS = 1800  # 30 minutes - if not seen for 30 min, auto clock-out
DEBOUNCE_FRAMES = 3
DEBOUNCE_SECONDS = 0.5             # time-based debounce for clock-in/recovery transitions
# How long a person must be continuously invisible before entering TEMP_LOST.
# Kept much higher than DEBOUNCE_SECONDS so brief occlusions (person leans over desk,
# walks behind a colleague, ByteTrack reassigns track_id) never trigger TEMP_LOST.
TEMP_LOST_DEBOUNCE_SECONDS = 30   # 30 seconds of continuous invisibility required

# Camera (no PTZ/movement control)
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", 0))
CAMERA_INDICES = [int(x) for x in os.getenv("CAMERA_INDICES", "0,1").split(",")]
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
CAMERA_FPS = 30  # match the capture rate to the main loop (no point reading at 60fps)

# Multi-Camera Handoff (see HANDOFF_WINDOW_SECONDS above in Tracking section)
HISTOGRAM_MATCH_THRESHOLD = 0.7   # cv2.compareHist correlation threshold

# Physical camera layout — used to score cross-camera handoff direction.
# cam 0 = left side of room, cam 1 = right side of room.
# A person exiting the RIGHT half of cam 0 should enter the LEFT half of cam 1,
# and vice versa.  Geometrically inconsistent entries get a score penalty.
CAMERA_LEFT_ID = int(os.getenv("CAMERA_LEFT_ID", 0))
CAMERA_RIGHT_ID = int(os.getenv("CAMERA_RIGHT_ID", 1))

# Audio Alerts
AUDIO_ENABLED = os.getenv("AUDIO_ENABLED", "true").lower() == "true"
AUDIO_WARNING_DELAY = 300         # seconds before first "missing employee" spoken warning (was 60)
AUDIO_WARNING_INTERVAL = 600      # seconds between repeated missing-person warnings (was 300)

# Notification cooldowns (prevent alert floods in main loop)
OVERTIME_ALERT_COOLDOWN = 3600    # seconds between repeated overtime alerts per employee
BREAK_ALERT_COOLDOWN = 3600       # seconds between repeated break-reminder alerts per employee
UNKNOWN_PERSON_AUDIO_COOLDOWN = 180  # seconds between "look at camera" beeps
# Minimum gap between consecutive TEMP_LOST and recovered Discord notifications
# per employee.  Avoids "Alice vanished / Alice is back" spam from brief absences.
PRESENCE_NOTIFY_COOLDOWN = 300    # 5 minutes

# Discord — 3-channel webhook URLs
DISCORD_WEBHOOK_HERE_GONE = os.getenv("DISCORD_WEBHOOK_HERE_GONE", "")
DISCORD_WEBHOOK_CLOCK_LOGS = os.getenv("DISCORD_WEBHOOK_CLOCK_LOGS", "")
DISCORD_WEBHOOK_ADMIN = os.getenv("DISCORD_WEBHOOK_ADMIN", "")
# Legacy fallback: if only one URL is set, use it for all channels
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

# Discord Bot (for reading the Schedules channel — separate from webhooks)
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")
DISCORD_SCHEDULES_CHANNEL_ID = os.getenv("DISCORD_SCHEDULES_CHANNEL_ID", "")

# Embed colours (Discord uses decimal, not hex)
DISCORD_COLOR_GREEN = 0x2ECC71    # clock-in, recovered
DISCORD_COLOR_RED = 0xE74C3C      # clock-out, errors
DISCORD_COLOR_ORANGE = 0xF39C12   # warnings, temp-lost
DISCORD_COLOR_BLUE = 0x3498DB     # info, summaries
DISCORD_COLOR_PURPLE = 0x9B59B6   # admin/security

# Paths
EMBEDDINGS_PATH = Path(os.getenv("EMBEDDINGS_PATH", BASE_DIR / "data" / "employees" / "embeddings.pkl"))
LOGS_DIR = Path(os.getenv("LOGS_DIR", BASE_DIR / "data" / "logs"))
SNAPSHOT_DIR = Path(os.getenv("SNAPSHOT_DIR", BASE_DIR / "data" / "snapshots"))

# Ensure directories exist
EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

# Security – manager password (SHA-256 hash of the password)
import hashlib as _hl
MANAGER_PASSWORD_HASH = os.getenv(
    "MANAGER_PASSWORD_HASH",
    "04773d38e9fd33afcf0da6ddb37875497de06cd6b383d3ec121db2ee4665c7b7",
)

# Work-hours expectations (for late/early alerts)
EXPECTED_START_HOUR = int(os.getenv("EXPECTED_START_HOUR", 9))   # 09:00
EXPECTED_START_MINUTE = int(os.getenv("EXPECTED_START_MINUTE", 0))  # :00
EXPECTED_END_HOUR = int(os.getenv("EXPECTED_END_HOUR", 17))     # 17:00
DAILY_OVERTIME_THRESHOLD = 8 * 3600    # 8 hours in seconds
WEEKLY_OVERTIME_THRESHOLD = 40 * 3600  # 40 hours in seconds

# Break detection
BREAK_MAX_SECONDS = 15 * 60  # TEMP_LOST < 15 min = "break"
