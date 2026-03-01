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
DETECTION_SKIP_FRAMES = 5  # run YOLO every N frames; tracker interpolates in between
ONNX_THREADS = 2  # threads for ONNX Runtime inference (limits CPU saturation)

# Face Recognition
FACE_SIMILARITY_THRESHOLD = 0.72  # Higher = stricter matching (raised from 0.65 to reduce false positives)
FACE_DETECTION_SIZE = (640, 640)
FACE_FRAME_MAX_DIM = 480  # resize long side before DeepFace to save time

# Tracking
TRACK_BUFFER = 150  # frames to keep lost tracks alive (~5s at 30fps)

# Persistent body tracking -- minimise face re-scanning
BINDING_PERSIST_SECONDS = 7200     # 2 hours: auto-bind via descriptor without face rec
SAME_CAM_HANDOFF_WINDOW = 300     # 5 minutes: same-camera re-ID window
SPATIAL_MATCH_PIXELS = 150         # px radius for position-based matching boost

# Track Validation (to prevent binding hands/objects as people)
MIN_TRACK_AREA = 3000  # minimum pixels^2 for a valid person track (lowered for distance)
TRACK_ASPECT_RATIO_MIN = 0.2  # width/height min (person is taller than wide)
TRACK_ASPECT_RATIO_MAX = 1.0  # width/height max (allow sitting/crouching)
FACE_IN_UPPER_RATIO = 0.7  # face center must be in upper 70% of track bbox
BINDING_IOU_THRESHOLD = 0.1  # lower threshold - face bbox is smaller than body bbox

# State Machine
TEMP_LOST_TIMEOUT_SECONDS = 1800  # 30 minutes - if not seen for 30 min, auto clock-out
DEBOUNCE_FRAMES = 3
DEBOUNCE_SECONDS = 0.5  # time-based debounce (preferred over frame-based)

# Camera (no PTZ/movement control)
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", 0))
CAMERA_INDICES = [int(x) for x in os.getenv("CAMERA_INDICES", "0,1").split(",")]
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
CAMERA_FPS = 30  # match the capture rate to the main loop (no point reading at 60fps)

# Multi-Camera Handoff
HANDOFF_WINDOW_SECONDS = 30       # seconds after losing a track to attempt histogram handoff
HISTOGRAM_MATCH_THRESHOLD = 0.7   # cv2.compareHist correlation threshold

# Audio Alerts
AUDIO_ENABLED = os.getenv("AUDIO_ENABLED", "true").lower() == "true"
AUDIO_WARNING_DELAY = 60          # seconds before first spoken warning
AUDIO_WARNING_INTERVAL = 300      # seconds between repeated warnings

# Discord — 3-channel webhook URLs
DISCORD_WEBHOOK_HERE_GONE = os.getenv("DISCORD_WEBHOOK_HERE_GONE", "")
DISCORD_WEBHOOK_CLOCK_LOGS = os.getenv("DISCORD_WEBHOOK_CLOCK_LOGS", "")
DISCORD_WEBHOOK_ADMIN = os.getenv("DISCORD_WEBHOOK_ADMIN", "")
# Legacy fallback: if only one URL is set, use it for all channels
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

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
