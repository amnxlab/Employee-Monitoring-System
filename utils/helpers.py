from datetime import datetime, timedelta
from typing import Tuple


def calculate_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """
    Calculate Intersection over Union between two bounding boxes.
    Boxes are in format (x1, y1, x2, y2).
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def get_box_center(box: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """Get center point of a bounding box (x1, y1, x2, y2)."""
    return ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def format_hours(seconds: float) -> str:
    """Format duration as decimal hours (e.g., 38.5 hrs)."""
    hours = seconds / 3600
    return f"{hours:.1f} hrs"


def get_week_number(dt: datetime = None) -> int:
    """Get ISO week number for a given date."""
    if dt is None:
        dt = datetime.now()
    return dt.isocalendar()[1]


def get_week_start_date(dt: datetime = None) -> datetime:
    """Get the Monday of the week for a given date."""
    if dt is None:
        dt = datetime.now()
    days_since_monday = dt.weekday()
    return dt - timedelta(days=days_since_monday)


def is_new_week(last_check: datetime, current: datetime = None) -> bool:
    """Check if we've crossed into a new week since last check."""
    if current is None:
        current = datetime.now()
    return get_week_number(last_check) != get_week_number(current)
