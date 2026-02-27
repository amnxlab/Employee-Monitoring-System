# Employee Monitoring System

Automated smart monitoring and employee attendance platform with identity recognition, continuous tracking, time logging, and Discord-based reporting.

## Features

- **Face Recognition**: InsightFace-based identity verification
- **Person Tracking**: ByteTrack multi-object tracking with ID persistence
- **PTZ Camera Control**: OBSBOT Tiny 2 integration for subject tracking
- **Attendance State Machine**: Automatic clock in/out with grace periods
- **Discord Integration**: Real-time notifications via webhooks
- **Weekly Reports**: Automated time tracking and summaries

## Requirements

- Python 3.9+
- NVIDIA GPU (recommended for real-time performance)
- OBSBOT Tiny 2 camera
- OBSBOT WebCam SDK (download from [OBSBOT website](https://www.obsbot.com/download))

## Installation

1. Clone or download this repository

2. Create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # or
   source venv/bin/activate  # Linux/Mac
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   **Face recognition (Windows):** The app can use either **InsightFace** or **DeepFace**. To avoid installing Microsoft C++ Build Tools, use DeepFace (no compilation):
   ```bash
   python -m pip install deepface
   ```
   Or run the helper script (tries InsightFace first, then installs DeepFace if that fails):
   ```bash
   python install_face_recognition.py
   ```
   If you prefer InsightFace, install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) first, then: `pip install insightface>=0.7.3`

4. Copy `.env.example` to `.env` and configure:
   ```bash
   copy .env.example .env
   ```

5. Edit `.env` with your Discord webhook URL:
   ```
   DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
   ```

6. Install OBSBOT WebCam SDK and place DLLs in system PATH or project directory

## Usage

### Register Employees

Before running the system, register employees:

```bash
python register_employee.py --id EMP001 --name "John Doe"
```

Follow the on-screen instructions to capture face embeddings.

### Run the Monitoring System

```bash
python main.py
```

### Keyboard Controls

- `q` - Quit the application
- `r` - Enter registration mode
- `d` - Toggle debug display

## Project Structure

```
monitoring_system/
├── main.py                    # Application entry point
├── config.py                  # Configuration settings
├── register_employee.py       # Employee registration utility
├── core/
│   ├── camera.py              # OBSBOT camera + PTZ control
│   ├── detector.py            # YOLOv8n person detection
│   ├── face_recognition.py    # InsightFace embedding + matching
│   ├── tracker.py             # ByteTrack integration
│   └── id_binder.py           # Track ID to Employee ID binding
├── attendance/
│   ├── state_machine.py       # Per-employee state machine
│   ├── timer.py               # Work time tracking
│   └── logger.py              # Weekly log file management
├── integrations/
│   └── discord_webhook.py     # Discord notifications
├── data/
│   ├── employees/             # Employee database
│   │   └── embeddings.pkl     # Stored face embeddings
│   └── logs/                  # Weekly attendance logs
└── utils/
    └── helpers.py             # Utility functions
```

## Configuration

Edit `config.py` to customize:

| Setting | Default | Description |
|---------|---------|-------------|
| `PERSON_CONFIDENCE_THRESHOLD` | 0.5 | YOLO detection confidence |
| `FACE_SIMILARITY_THRESHOLD` | 0.6 | Face matching threshold |
| `TEMP_LOST_TIMEOUT_SECONDS` | 600 | Grace period before auto clock-out |
| `DEBOUNCE_FRAMES` | 3 | Frames required for state change |
| `PTZ_ENABLED` | True | Enable camera PTZ tracking |

## Attendance States

```
OUT → DETECTED → CLOCKED_IN → TEMP_LOST → CLOCKED_OUT
                     ↑            │
                     └────────────┘ (re-identified within 10 min)
```

## Discord Notifications

The system sends the following notifications to your configured Discord channel:

- **Clock In**: `[Name] clocked in at HH:MM`
- **Clock Out**: `[Name] clocked out at HH:MM` with session duration
- **Weekly Summary**: Posted Sunday night with total hours per employee

## Privacy Notice

This system uses facial recognition technology. Ensure you have:
- Obtained consent from all monitored employees
- Complied with local privacy regulations (GDPR, etc.)
- Secured the embeddings database appropriately

## Troubleshooting

### Camera not detected
- Check USB connection
- Verify CAMERA_INDEX in .env matches your camera
- Install OBSBOT WebCam software

### Low FPS
- Ensure GPU is being used (check CUDA availability)
- Reduce FRAME_WIDTH/FRAME_HEIGHT in config.py
- Close other GPU-intensive applications

### Face not recognized
- Re-register the employee with better lighting
- Lower FACE_SIMILARITY_THRESHOLD (may increase false positives)
- Ensure face is clearly visible to camera

## License

Internal use only. All rights reserved.
