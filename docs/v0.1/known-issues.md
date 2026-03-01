# Known Issues & Discrepancies — v0.1

This document records all confirmed bugs, stale documentation, design inconsistencies, and code smells found during the v0.1 audit. Each item is a candidate fix for subsequent work.

---

## Documentation / README Discrepancies

### IS-001 — README describes removed features
**Severity:** Low (misleading docs, not a runtime bug)  
**File:** `README.md`

The README mentions:
- PTZ camera control (`PTZ_ENABLED`, OBSBOT WebCam SDK, DLL installation)
- `PTZ_ENABLED` as a config table entry

**Reality:** `core/ptz_controller.py` is a no-op stub. `core/camera.py` explicitly states "All PTZ/movement control has been removed." `PTZ_ENABLED` does not exist in `config.py`.

**Fix:** Remove PTZ references from README; document the stub.

---

### IS-002 — README config table has wrong defaults
**Severity:** Low  
**File:** `README.md`

| Setting | README says | `config.py` actual |
|---------|-------------|-------------------|
| `FACE_SIMILARITY_THRESHOLD` | `0.6` | `0.72` |
| `TEMP_LOST_TIMEOUT_SECONDS` | `600` (10 min) | `1800` (30 min) |
| `PTZ_ENABLED` | `True` | **does not exist** |

**Fix:** Update README config table to match `config.py`.

---

### IS-003 — `.env.example` missing from repository
**Severity:** Medium (onboarding friction)  
**File:** `.env.example` (missing)

The README instructs users to `copy .env.example .env`, but no `.env.example` file exists in the repository.

**Fix:** Create `.env.example` with all relevant environment variable names and placeholder values. See [configuration.md](configuration.md) for the complete list.

---

### IS-004 — `check_storage.py` is developer-local utility committed to repo
**Severity:** Low (noise)  
**File:** `check_storage.py`

The file hard-codes paths for a specific developer's machine (`C:\Users\Amin-PC\…`). It is not a general-purpose utility.

**Fix:** Either generalise it (use `Path.home()`) or remove it from the repository and add to `.gitignore`.

---

## Architecture / Design Issues

### IS-005 — Dual logging systems running in parallel
**Severity:** Medium (redundancy, disk waste)  
**Files:** `attendance/logger.py`, `attendance/event_logger.py`

Both loggers are active simultaneously:
- `AttendanceLogger` (legacy) writes `.log` text files: `{EMP}_{YEAR}_W{WW}.log`
- `EventLogger` (current) writes `.json` files: `{EMP}_{YEAR}_W{WW}.json`

There is no reads of the `.log` files anywhere in the codebase — they serve no runtime purpose.

**Fix:** Deprecate and remove `AttendanceLogger`; rely solely on `EventLogger`.

---

### IS-006 — `CAMERA_INDICES` default includes camera index 1 which may not exist
**Severity:** Medium (causes startup error on single-camera systems)  
**File:** `config.py` line: `CAMERA_INDICES = [int(x) for x in os.getenv("CAMERA_INDICES", "0,1").split(",")]`

The default value of `"0,1"` means the system always tries to open two cameras. On a machine with only one camera (index 0), `CameraManager` logs an error for camera 1 but continues with camera 0. However, this is unexpected default behaviour for first-time setup.

**Fix:** Change the default to `"0"` and document multi-camera setup as an explicit opt-in configuration.

---

### IS-007 — `AttendanceStateMachine` new session after `CLOCKED_OUT` resets all timers
**Severity:** Medium (potential double-count or data loss edge case)  
**File:** `attendance/state_machine.py` — `_handle_clocked_out_state()`

When an employee is seen again after being `CLOCKED_OUT`, the FSM transitions `CLOCKED_OUT → DETECTED → CLOCKED_IN`. The `_handle_detected_state()` resets `session_seconds = 0.0` and sets a new `clock_in_time`. This is correct for a new session.

However, if `MonitoringSystem` fails to call `TimerManager.clock_out()` before the state machine re-enters `DETECTED`, the old session's time could be lost or double-counted in the `TimerManager`.

**Fix:** Audit `MonitoringSystem._handle_clock_in()` to ensure it always calls `TimerManager.clock_out()` for any employee already in `CLOCKED_IN` or `TEMP_LOST` before starting a new session timer.

---

### IS-008 — `_FaceRecWorker` queue drop silently skips camera frames
**Severity:** Low (by design, but undocumented)  
**File:** `main.py` — `_FaceRecWorker.submit()`

The queue has `maxsize=1`. If the face recognition thread is still processing frame N, submitting frame N+1 silently drops it by calling `_q.get_nowait()` first. This is a valid performance trade-off, but means face recognition may be skipped for several seconds if inference is slow.

**No fix needed**, but this should be acknowledged in code comments. Consider logging when a frame is dropped at DEBUG level.

---

### IS-009 — `pyttsx3` audio runs synchronously on some platforms
**Severity:** Low  
**File:** `core/audio_alert.py`

On Windows, `pyttsx3` with the SAPI5 driver can block the calling thread. The code starts it on a background thread, but the engine's `runAndWait()` call may still accumulate a backlog if many alerts are queued quickly.

**Fix:** Add a max-queue-depth guard (e.g., drop new alerts if more than 2 are pending) to avoid audio falling minutes behind actual events.

---

## Missing Features Referenced in Code

### IS-010 — `BREAK_MAX_SECONDS` is defined but not used
**Severity:** Low  
**File:** `config.py`

`BREAK_MAX_SECONDS = 15 * 60` is defined (and documented: "TEMP_LOST < 15 min = break"), but no code in `EventLogger`, `TimerManager`, or `DiscordNotifier` distinguishes between a "break" and an "absence" TEMP_LOST event.

**Fix:** In `EventLogger.log_clock_out()` or `DiscordNotifier.notify_clock_out()`, check the sum of `interruption` durations against `BREAK_MAX_SECONDS` and label accordingly.

---

### IS-011 — Weekly overtime threshold defined but no alert fired
**Severity:** Low  
**File:** `config.py` — `WEEKLY_OVERTIME_THRESHOLD = 40 * 3600`

`DAILY_OVERTIME_THRESHOLD` is checked (an audio alert for >8 h/day), but `WEEKLY_OVERTIME_THRESHOLD` is not used anywhere to fire a Discord or audio alert for >40 h/week.

**Fix:** Add a weekly overtime check in the weekly report generation or in `TimerManager.clock_out()`.

---

### IS-012 — `SNAPSHOT_DIR` auto-cleanup not scheduled
**Severity:** Low  
**File:** `core/snapshot.py` — `cleanup_old_snapshots()`

`cleanup_old_snapshots()` is imported in `main.py` but it is unclear whether it is actually called on a schedule. Snapshots will accumulate indefinitely without cleanup.

**Fix:** Verify the cleanup is wired into the `schedule` jobs or add a daily cleanup task.

---

## Dependency & Installation Issues

### IS-013 — `requirements.txt` does not pin `pyttsx3`
**Severity:** Low  
**File:** `requirements.txt`

`pyttsx3` is used in `core/audio_alert.py` but is not listed in `requirements.txt`. First-time users will encounter an `ImportError` if audio is enabled (default: `AUDIO_ENABLED=true`) and `pyttsx3` is not installed.

**Fix:** Add `pyttsx3>=2.90` to `requirements.txt` (or add clear install instructions, noting it is optional).

---

### IS-014 — `onnxruntime-gpu` pinned in requirements but GPU may not be available
**Severity:** Low  
**File:** `requirements.txt` — `onnxruntime-gpu>=1.16.0`

On CPU-only machines, `onnxruntime-gpu` installs successfully but runs on CPU anyway. This is harmless but confusing. For machines without CUDA, `onnxruntime` (CPU build) is slightly smaller.

**No fix required** unless package size is a concern. Consider providing two requirements files or conditional install instructions.

---

### IS-015 — `insightface` not in `requirements.txt`
**Severity:** Low  
**File:** `requirements.txt`

`deepface` is listed as the default, but no `insightface` entry exists (not even commented out correctly — there are two comment lines but the actual `insightface` package line is missing from the installable section).

**Fix:** Add `# insightface>=0.7.3  # optional, needs C++ Build Tools on Windows` as a comment in `requirements.txt`.

---

## Security

### IS-016 — Default manager password is hard-coded and public
**Severity:** High  
**File:** `config.py`

```python
MANAGER_PASSWORD_HASH = os.getenv(
    "MANAGER_PASSWORD_HASH",
    "04773d38e9fd33afcf0da6ddb37875497de06cd6b383d3ec121db2ee4665c7b7",
)
```

The default hash is committed to the public repository. Anyone can reverse-lookup or brute-force this common hash. Any deployment using the default password with the watchdog running is thus unprotected.

**Fix:**
1. Remove the default fallback hash from `config.py` (require explicit configuration).
2. On startup, if `MANAGER_PASSWORD_HASH` is not set (or matches the default), log a prominent warning and refuse to start the watchdog in protected mode.
3. Update the `.env.example` to include clear instructions for setting a custom password.

---

## Summary Table

| ID | Category | Severity | File(s) | Status |
|----|----------|----------|---------|--------|
| IS-001 | Docs | Low | README.md | Open |
| IS-002 | Docs | Low | README.md | Open |
| IS-003 | Docs | Medium | `.env.example` | Open |
| IS-004 | Code quality | Low | check_storage.py | Open |
| IS-005 | Architecture | Medium | logger.py, event_logger.py | Open |
| IS-006 | Config | Medium | config.py | Open |
| IS-007 | Logic | Medium | state_machine.py, main.py | Open |
| IS-008 | Design | Low | main.py | Acknowledged |
| IS-009 | Audio | Low | audio_alert.py | Open |
| IS-010 | Missing feature | Low | config.py | Open |
| IS-011 | Missing feature | Low | config.py | Open |
| IS-012 | Missing feature | Low | snapshot.py, main.py | Open |
| IS-013 | Dependencies | Low | requirements.txt | Open |
| IS-014 | Dependencies | Low | requirements.txt | Info |
| IS-015 | Dependencies | Low | requirements.txt | Open |
| IS-016 | Security | **High** | config.py | Open |
