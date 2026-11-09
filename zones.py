import json
import os
import cv2

from settings import (
    ZONES_FILE,
    RESTRICTED_COLOR,
    WARNING_COLOR,
    FONT
)


def save_zones(zones):
    with open(ZONES_FILE, "w", encoding="utf-8") as file:
        json.dump(zones, file, indent=2)


def _is_old_default(zone):
    return (
        zone.get("name") == "Zone 1"
        and zone.get("type", "Restricted") == "Restricted"
        and abs(zone.get("x1", -1) - 0.5) < 1e-6
        and abs(zone.get("y1", -1) - 0.0) < 1e-6
        and abs(zone.get("x2", -1) - 1.0) < 1e-6
        and abs(zone.get("y2", -1) - 1.0) < 1e-6
        and zone.get("timer", 10) == 10
    )


def load_zones():
    if not os.path.exists(ZONES_FILE):
        return []

    try:
        with open(ZONES_FILE, encoding="utf-8") as file:
            zones = json.load(file)
    except (json.JSONDecodeError, OSError):
        return []

    if not isinstance(zones, list):
        return []

    kept_zones = [
        zone for zone in zones
        if not _is_old_default(zone)
    ]

    if len(kept_zones) != len(zones):
        save_zones(kept_zones)

    for zone in kept_zones:
        zone.setdefault("enabled", True)
        zone.setdefault("timer", 10)
        zone.setdefault("type", "Restricted")

    return kept_zones


def zone_color(zone):
    if zone["type"] == "Restricted":
        return RESTRICTED_COLOR

    return WARNING_COLOR


def draw_zone_box(img, zone, width, height, active=True):
    x1 = int(width * zone["x1"])
    y1 = int(height * zone["y1"])
    x2 = int(width * zone["x2"])
    y2 = int(height * zone["y2"])

    if not zone.get("enabled", True):
        color = (150, 150, 150)
    elif active:
        color = zone_color(zone)
    else:
        color = (255, 255, 255)

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)

    cv2.putText(
        img,
        f"{zone['name']} - {zone['type']}",
        (x1 + 5, y1 + 18),
        FONT,
        0.5,
        color,
        1
    )