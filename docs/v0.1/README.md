# Employee Monitoring System — v0.1 Documentation

> **Snapshot date:** 2026-02-28  
> **Branch:** `main` (`amnxlab/Employee-Monitoring-System`)  
> **Python:** 3.9+  
> **Platform:** Windows (primary), Linux/macOS (partial)

---

## What This System Does

An automated, camera-based employee attendance platform that:

1. Reads live video from one or more webcams.
2. Detects every person in frame using a YOLOv8n model (ONNX-accelerated).
3. Tracks each person across frames with ByteTrack (persistent IDs).
4. Identifies known employees via face recognition (InsightFace or DeepFace).
5. Runs a per-employee attendance state machine (clock-in → temp-lost → clock-out).
6. Accumulates and persists session times in structured JSON logs.
7. Sends real-time notifications to Discord (3-channel architecture).
8. Speaks audio alerts via text-to-speech (pyttsx3).
9. Auto-restarts on crash via `watchdog.py`.

---

## Quick Start

```bash
# 1. Create & activate virtual environment
python -m venv venv
venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment (copy and edit .env)
copy .env.example .env   # NOTE: .env.example not yet committed – see Configuration docs

# 4. Register at least one employee
python register_employee.py --id EMP001 --name "Alice Smith"

# 5. Run the system
python main.py

# 5b. Or run with auto-restart watchdog
python watchdog.py
```

---

## Repository Layout

```
Employee-Monitoring-System/
├── main.py                     # Entry point & MonitoringSystem orchestrator
├── config.py                   # All tunable parameters (reads .env)
├── watchdog.py                 # Auto-restart daemon
├── register_employee.py        # CLI tool: add / remove / verify employees
├── check_storage.py            # Utility: inspect stored embeddings / logs
├── install_face_recognition.py # Helper: installs InsightFace or DeepFace
├── requirements.txt            # Pip dependencies
├── run.bat                     # Windows convenience launcher
├── FUTURE.md                   # Roadmap / planned features
│
├── core/                       # Computer-vision pipeline
│   ├── __init__.py
│   ├── camera.py               # UVC webcam wrapper (OpenCV)
│   ├── camera_manager.py       # Multi-camera threaded manager
│   ├── detector.py             # YOLOv8n person detection (ONNX / PyTorch)
│   ├── tracker.py              # ByteTrack multi-object tracker
│   ├── face_recognition.py     # Face detection + embedding + matching
│   ├── id_binder.py            # Per-camera track-ID ↔ employee-ID binding
│   ├── global_id_binder.py     # Cross-camera persistent binding + body descriptor
│   ├── audio_alert.py          # TTS alerts (pyttsx3)
│   ├── ptz_controller.py       # (Legacy/unused) PTZ camera control stub
│   └── snapshot.py             # JPEG snapshot capture with watermark
│
├── attendance/                 # Attendance logic
│   ├── __init__.py
│   ├── state_machine.py        # Per-employee FSM (OUT → CLOCKED_IN → TEMP_LOST …)
│   ├── timer.py                # Session / daily / weekly time accumulation
│   ├── logger.py               # Legacy text-based .log writer
│   └── event_logger.py         # JSON-based session logger (current)
│
├── integrations/               # External service connectors
│   ├── __init__.py
│   └── discord_webhook.py      # 3-channel Discord webhook notifier
│
├── utils/                      # Shared helpers
│   ├── __init__.py
│   └── helpers.py              # IoU, duration formatting, week numbers
│
└── data/
    ├── employees/
    │   └── embeddings.pkl      # Serialised face embeddings (binary, git-ignored)
    └── logs/
        └── {EMP}_{YEAR}_W{WW}.json   # Weekly JSON attendance logs
```

---

## Currently Registered Employees (as of snapshot)

Log files found in `data/logs/`:

| Employee ID | Week Log File              |
|-------------|----------------------------|
| EMP001      | EMP001_2026_W09.json       |
| EMP002      | EMP002_2026_W09.json       |
| EMP005      | EMP005_2026_W09.json       |
| EMP006      | EMP006_2026_W09.json       |

---

## Key Keyboard Controls (during runtime)

| Key | Action |
|-----|--------|
| `q` | Initiate quit (prompts manager password) |
| `r` | Enter live registration mode |
| `d` | Toggle debug overlay |

---

## See Also

- [architecture.md](architecture.md) — Component design and threading model
- [modules.md](modules.md) — Per-file module reference
- [configuration.md](configuration.md) — Full `config.py` parameter reference
- [data-flow.md](data-flow.md) — End-to-end processing pipeline
- [known-issues.md](known-issues.md) — Confirmed bugs and discrepancies
