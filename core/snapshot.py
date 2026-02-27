"""
Snapshot capture for clock-in events.

Saves a camera frame with the event ID watermarked on the image and the
person's bounding box highlighted.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)

# Default snapshot directory
SNAPSHOT_DIR = Path(getattr(config, "SNAPSHOT_DIR", config.BASE_DIR / "data" / "snapshots"))
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)


def save_snapshot(
    frame: np.ndarray,
    event_id: str,
    person_bbox: Optional[Tuple[int, int, int, int]] = None,
    snapshot_dir: Path = None,
) -> Optional[str]:
    """
    Save a snapshot frame with the event ID watermarked on the image.

    Args:
        frame: The full camera frame (BGR).
        event_id: The event ID to watermark.
        person_bbox: Optional (x1, y1, x2, y2) bounding box to highlight.
        snapshot_dir: Override the default snapshot directory.

    Returns:
        Relative path to the saved snapshot, or None on failure.
    """
    out_dir = snapshot_dir or SNAPSHOT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{event_id}.jpg"
    filepath = out_dir / filename

    try:
        img = frame.copy()

        # Draw person bounding box
        if person_bbox is not None:
            x1, y1, x2, y2 = person_bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Watermark: event ID in the bottom-left corner
        h, w = img.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        text = event_id

        # Measure text size for background rectangle
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        pad = 8
        text_x = pad
        text_y = h - pad

        # Semi-transparent dark background
        overlay = img.copy()
        cv2.rectangle(
            overlay,
            (text_x - pad, text_y - th - pad),
            (text_x + tw + pad, text_y + baseline + pad),
            (0, 0, 0),
            cv2.FILLED,
        )
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

        # Draw text
        cv2.putText(
            img, text, (text_x, text_y),
            font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
        )

        # Also add timestamp top-right
        from datetime import datetime
        ts_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        (tsw, tsh), _ = cv2.getTextSize(ts_text, font, 0.5, 1)
        ts_x = w - tsw - pad
        ts_y = tsh + pad
        cv2.putText(
            img, ts_text, (ts_x, ts_y),
            font, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )

        # Save (quality 75 → ~30% smaller than 90, minimal visual difference)
        cv2.imwrite(str(filepath), img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        logger.info(f"Snapshot saved: {filepath}")

        return str(filepath)

    except Exception as e:
        logger.error(f"Failed to save snapshot: {e}")
        return None


def cleanup_old_snapshots(max_age_days: int = 30, snapshot_dir: Path = None):
    """Delete snapshot files older than *max_age_days*.

    Called from the daily summary routine so the directory stays bounded.
    """
    import time as _t
    out_dir = snapshot_dir or SNAPSHOT_DIR
    cutoff = _t.time() - max_age_days * 86400
    removed = 0
    for f in out_dir.iterdir():
        if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png"):
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
                    removed += 1
            except OSError:
                pass
    if removed:
        logger.info(f"Snapshot cleanup: removed {removed} file(s) older than {max_age_days} days")
