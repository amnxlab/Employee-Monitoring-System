"""
One-time migration script to fix existing JSON log data.

Fixes:
  1. Close orphaned sessions (closed=false) with best-guess timestamps.
  2. Normalize absolute snapshot paths to relative filenames.
  3. Reconcile daily_totals with actual closed-session durations.
  4. Reconcile weekly_total_seconds from daily_totals.

Usage:
    python migrate_logs.py              # dry-run (preview changes)
    python migrate_logs.py --apply      # apply changes in-place

Backups are created automatically before modifying any file.
"""

import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

LOGS_DIR = Path(__file__).parent / "data" / "logs"


def migrate_log_file(path: Path, apply: bool = False) -> dict:
    """Migrate a single JSON log file. Returns a summary dict."""
    summary = {
        "file": path.name,
        "orphaned_closed": 0,
        "paths_normalized": 0,
        "daily_totals_fixed": 0,
        "weekly_total_fixed": False,
    }

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    changed = False

    # ── 1. Close orphaned sessions ──
    for session in data.get("sessions", []):
        if not session.get("closed", False):
            # Determine best closing timestamp
            last_ts = session.get("clock_in", "")
            interruptions = session.get("interruptions", [])
            if interruptions:
                last_intr = interruptions[-1]
                last_ts = (
                    last_intr.get("recovered_at")
                    or last_intr.get("lost_at")
                    or last_ts
                )
                # Close any open interruptions
                for intr in interruptions:
                    if intr.get("recovered_at") is None:
                        intr["recovered_at"] = last_ts
                        changed = True

            session["clock_out"] = last_ts
            session["closed"] = True
            # We can't know the real duration — leave as recorded or 0
            if session.get("duration_seconds", 0.0) == 0.0:
                # Try to compute from clock_in to clock_out minus interruptions
                try:
                    ci = datetime.fromisoformat(session["clock_in"])
                    co = datetime.fromisoformat(last_ts)
                    raw_secs = (co - ci).total_seconds()
                    # Subtract interruption durations
                    lost_secs = 0.0
                    for intr in interruptions:
                        try:
                            lt = datetime.fromisoformat(intr["lost_at"])
                            rt = datetime.fromisoformat(intr.get("recovered_at", intr["lost_at"]))
                            lost_secs += (rt - lt).total_seconds()
                        except (ValueError, KeyError):
                            pass
                    session["duration_seconds"] = max(0.0, raw_secs - lost_secs)
                except (ValueError, KeyError):
                    session["duration_seconds"] = 0.0

            summary["orphaned_closed"] += 1
            changed = True

    # ── 2. Normalize snapshot paths ──
    for session in data.get("sessions", []):
        snap = session.get("snapshot_path")
        if snap and (os.sep in snap or "/" in snap or "\\" in snap):
            # Extract just the filename
            filename = Path(snap).name
            session["snapshot_path"] = filename
            summary["paths_normalized"] += 1
            changed = True

    # ── 3. Reconcile daily_totals from closed sessions ──
    computed_daily = {}
    for session in data.get("sessions", []):
        if session.get("closed"):
            day = session.get("date_str", "")
            if day:
                computed_daily[day] = computed_daily.get(day, 0.0) + session.get("duration_seconds", 0.0)

    existing_daily = data.get("daily_totals", {})
    if computed_daily != existing_daily:
        data["daily_totals"] = computed_daily
        summary["daily_totals_fixed"] = len(computed_daily)
        changed = True

    # ── 4. Reconcile weekly total ──
    new_weekly = sum(computed_daily.values())
    if data.get("weekly_total_seconds", 0.0) != new_weekly:
        data["weekly_total_seconds"] = new_weekly
        summary["weekly_total_fixed"] = True
        changed = True

    if changed and apply:
        # Create backup
        backup = path.with_suffix(".json.bak")
        if not backup.exists():
            shutil.copy2(path, backup)
        # Write atomically
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(str(tmp), str(path))

    return summary


def main():
    apply = "--apply" in sys.argv

    if not LOGS_DIR.exists():
        print(f"Logs directory not found: {LOGS_DIR}")
        return

    log_files = sorted(LOGS_DIR.glob("*.json"))
    if not log_files:
        print("No JSON log files found.")
        return

    print(f"{'APPLYING' if apply else 'DRY RUN'}: Processing {len(log_files)} log file(s)...\n")

    total_orphaned = 0
    total_paths = 0
    total_daily = 0

    for path in log_files:
        try:
            summary = migrate_log_file(path, apply=apply)
        except Exception as e:
            print(f"  ERROR {path.name}: {e}")
            continue

        total_orphaned += summary["orphaned_closed"]
        total_paths += summary["paths_normalized"]
        total_daily += summary["daily_totals_fixed"]

        changes = []
        if summary["orphaned_closed"]:
            changes.append(f"{summary['orphaned_closed']} session(s) closed")
        if summary["paths_normalized"]:
            changes.append(f"{summary['paths_normalized']} path(s) normalized")
        if summary["daily_totals_fixed"]:
            changes.append(f"{summary['daily_totals_fixed']} day(s) reconciled")
        if summary["weekly_total_fixed"]:
            changes.append("weekly total updated")

        if changes:
            print(f"  {path.name}: {', '.join(changes)}")
        else:
            print(f"  {path.name}: OK (no changes needed)")

    print(f"\n{'=' * 50}")
    print(f"Total: {total_orphaned} sessions closed, {total_paths} paths normalized, {total_daily} daily totals fixed")
    if not apply and (total_orphaned or total_paths or total_daily):
        print("\nRun with --apply to write changes:")
        print(f"  python {Path(__file__).name} --apply")


if __name__ == "__main__":
    main()
