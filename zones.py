"""Zone persistence and drawing helpers for Digital Eye.

Camera 1 remains backward compatible with the existing zones.json file.
Cameras 2-4 are stored in zones_camera_2.json, zones_camera_3.json,
and zones_camera_4.json in the project directory.

Existing calls such as load_zones() and save_zones(zones) continue to work
and refer to Camera 1.
"""

import cv2

from settings import ZONES_FILE, RESTRICTED_COLOR, WARNING_COLOR, FONT
from camera_zones import load_camera_zones, save_camera_zones


def load_zones(camera_id=1):
    """Load zones for a camera. Defaults to Camera 1 for compatibility."""
    return load_camera_zones(camera_id)


def save_zones(zones, camera_id=1):
    """Save zones for a camera. Defaults to Camera 1 for compatibility."""
    return save_camera_zones(camera_id, zones)


def zone_color(zone):
    """Return the OpenCV BGR color for a zone."""
    zone_type = str(zone.get("type", "Restricted")).strip().lower()
    return WARNING_COLOR if zone_type == "warning" else RESTRICTED_COLOR


def draw_zone_box(frame, zone, active=False, thickness=2):
    """Draw a normalized zone rectangle and its label onto a BGR frame."""
    if frame is None or not isinstance(zone, dict):
        return frame

    height, width = frame.shape[:2]
    try:
        x1 = int(float(zone.get("x1", 0)) * width)
        y1 = int(float(zone.get("y1", 0)) * height)
        x2 = int(float(zone.get("x2", 0)) * width)
        y2 = int(float(zone.get("y2", 0)) * height)
    except (TypeError, ValueError):
        return frame

    color = zone_color(zone)
    if not zone.get("enabled", True):
        color = (130, 130, 130)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    label = str(zone.get("name", "Zone"))
    if not zone.get("enabled", True):
        label += " (Disabled)"
    elif active:
        label += " (ACTIVE)"

    text_y = max(18, y1 - 7)
    cv2.putText(frame, label, (x1, text_y), FONT, 0.55, color, 2, cv2.LINE_AA)
    return frame


__all__ = ["load_zones", "save_zones", "zone_color", "draw_zone_box"]
