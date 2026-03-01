# Data Flow — v0.1

End-to-end walkthrough of how raw camera frames become attendance records.

---

## Frame Lifecycle (per camera, per frame)

```
Camera hardware
      │  BGR frame (1280×720)
      ▼
_CameraThread._read_loop()
      │  stores latest_frame (thread-safe)
      ▼
MonitoringSystem._run_loop()
      │  calls CameraManager.get_latest_frames()
      │  iterates per camera
      ▼
_process_camera(cam_id, frame)
      │
      ├─ [every N frames - DETECTION_SKIP_FRAMES=5]
      │       PersonDetector.detect(frame)
      │       → resize frame to YOLO_IMGSZ=320
      │       → ONNX Runtime inference (2 threads)
      │       → NMS → filter class=0, conf≥0.5
      │       → List[Detection] (bbox, confidence)
      │       → cache as _prev_dets, _prev_scores
      │
      ├─ [every frame]
      │       PersonTracker.update(_prev_dets, _prev_scores)
      │       → sv.ByteTrack.update_with_detections()
      │       → List[Track] (track_id, bbox, status)
      │
      └─ return tracks
```

---

## Identity Resolution

```
List[Track] + frame
      │
      ├─ GlobalIDBinder.get_employee_id(cam_id, track_id)
      │     ├─ Check IDBinder cache (direct track→emp mapping)
      │     ├─ try_same_cam_reid() — body descriptor match
      │     └─ try_handoff() — cross-camera histogram match
      │
      ├─ [unbound tracks only, submitted to face-rec thread]
      │     _FaceRecWorker.submit(cam_id, frame, tracks)
      │
      └─ [face-rec thread - async]
            FaceRecognizer.detect_faces(frame)
            → resize long side to FACE_FRAME_MAX_DIM=480
            → InsightFace (or DeepFace) detection
            → List[FaceDetection] (bbox, embedding, confidence)

            FaceRecognizer.identify_all(faces)
            → cosine_similarity(face.embedding, employee.embedding)
            → filter by FACE_SIMILARITY_THRESHOLD=0.72
            → Dict[face_idx → employee_id]

            GlobalIDBinder.bind(cam_id, track_id, employee_id, frame, bbox)
            → IDBinder.bind(track_id, employee_id)
            → compute_body_descriptor(frame, bbox) → BodyDescriptor
            → store for 2-hour persistence window
```

---

## Attendance State Updates

```
For each employee_id seen/not-seen this frame:
      │
      ▼
AttendanceStateMachine.update(is_visible=True/False)
      │
      ├─ [transition: OUT → DETECTED → CLOCKED_IN]
      │       TimerManager.clock_in(emp_id)
      │       EventLogger.log_clock_in(emp_id, ...) → event_id
      │       save_snapshot(frame, event_id, bbox) → snapshot_path
      │       DiscordNotifier.notify_clock_in(...)  → #clock-in-logs
      │       DiscordNotifier.notify_arrived(...)   → #here-gone
      │       AudioAlertManager.play("clock_in" or "clock_in_late")
      │
      ├─ [transition: CLOCKED_IN → TEMP_LOST]
      │       EventLogger.log_temp_lost(emp_id)
      │       DiscordNotifier.notify_temp_lost(...)  → #here-gone
      │       AudioAlertManager.play("missing")
      │
      ├─ [transition: TEMP_LOST → CLOCKED_IN (recovered)]
      │       EventLogger.log_recovered(emp_id)
      │       DiscordNotifier.notify_recovered(...)  → #here-gone
      │       AudioAlertManager.play("recovered")
      │
      └─ [transition: TEMP_LOST → CLOCKED_OUT (timeout)]
              TimerManager.clock_out(emp_id, deduct_seconds=TEMP_LOST_TIMEOUT)
              EventLogger.log_clock_out(emp_id, duration) → saves JSON
              DiscordNotifier.notify_clock_out(...)  → #clock-in-logs
              DiscordNotifier.notify_left(...)       → #here-gone
              AudioAlertManager.play("clock_out")
```

---

## JSON Log Structure

Written to `data/logs/{EMP_ID}_{YEAR}_W{WW}.json`:

```json
{
  "employee_id": "EMP001",
  "employee_name": "Alice Smith",
  "year": 2026,
  "week": 9,
  "weekly_total_seconds": 144239.0,
  "daily_totals": {
    "2026-02-23": 28800.0,
    "2026-02-24": 27900.0
  },
  "sessions": [
    {
      "event_id": "EVT-20260223-091532-AB3X",
      "employee_id": "EMP001",
      "employee_name": "Alice Smith",
      "clock_in": "2026-02-23T09:15:32",
      "clock_out": "2026-02-23T17:15:32",
      "duration_seconds": 28800.0,
      "date_str": "2026-02-23",
      "snapshot_path": "data/snapshots/EVT-20260223-091532-AB3X.jpg",
      "interruptions": [
        {
          "lost_at": "2026-02-23T12:00:05",
          "recovered_at": "2026-02-23T12:08:42",
          "event_id_lost": "EVT-20260223-120005-XY9Z",
          "event_id_recovered": "EVT-20260223-120842-LM2P"
        }
      ],
      "closed": true
    }
  ]
}
```

---

## Weekly Report Trigger

```
schedule library (main thread, checked in _run_loop)
      │  every Sunday at 23:55
      ▼
MonitoringSystem._send_weekly_report()
      │
      ├─ TimerManager.get_weekly_total(emp_id) per employee
      ├─ EventLogger.generate_weekly_report_embed(totals, names)
      └─ DiscordNotifier.notify_weekly_report(embed) → #clock-in-logs
```

---

## Employee Registration Data Flow

```
register_employee.py --id EMP001 --name "Alice"
      │
      ▼
Camera.read() → display frame loop
      │  [user presses SPACE × 5]
      ▼
FaceRecognizer.detect_faces(frame)
      → List[FaceDetection]

[average 5 embeddings]
      │
      ▼
EmployeeDatabase.add_employee("EMP001", "Alice", avg_embedding)
      → pickle.dump → data/employees/embeddings.pkl
```

---

## Cross-Camera Handoff Flow

```
Employee last seen on Camera 0 at t=0
      │  TEMP_LOST on Camera 0
      │
      ▼  [within HANDOFF_WINDOW_SECONDS=30]

New unbound track appears on Camera 1 at t=15
      │
      ▼
GlobalIDBinder.try_handoff(cam_id=1, track_id=X, frame, bbox)
      │
      ├─ compute_histogram(body_crop_from_cam1)
      ├─ compare with stored histograms for all TEMP_LOST employees
      └─ if correlation ≥ HISTOGRAM_MATCH_THRESHOLD=0.7
              → bind(cam_id=1, track_id=X, emp_id)
              → TEMP_LOST → CLOCKED_IN transition (session continues)
```
