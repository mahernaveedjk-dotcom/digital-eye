"""
Fire and smoke detection for Digital Eye.

This file is separate from dashboard.py on purpose. You can change or replace the
fire/smoke feature here without touching the dashboard. dashboard.py only uses:

    FireSmokeDetector.load()        loads the AI model (downloaded once, then cached)
    detector.detect(frame, conf)    finds fire and smoke in one camera frame
    FireSmokeAlerts                 decides when an alert is due (confirmation, repeats)
    draw_fire_smoke(frame, dets)    draws boxes and a banner on the live picture
    build_email(alert, ...)         the subject and text of the alert email

Alert rules:  FIRE = CRITICAL,  SMOKE = WARNING.
Model: rabahdev/fire-smoke-yolov8n on Hugging Face (YOLOv8n trained on the D-Fire dataset,
license AGPL-3.0). It can miss fire/smoke and it can be fooled by look-alikes such as
orange lamps, steam, or fog, so treat it as a helper, never as a replacement for a real
smoke alarm.
"""
import os

import cv2

FIRE_SMOKE_REPO = "rabahdev/fire-smoke-yolov8n"
FIRE_SMOKE_FILE = "best.pt"
LOCAL_MODEL = os.path.join("models", "fire_smoke_best.pt")   # optional manual copy

FONT = cv2.FONT_HERSHEY_SIMPLEX
FIRE_COLOR = (0, 69, 255)        # orange-red (BGR)
SMOKE_COLOR = (190, 190, 190)    # light gray (BGR)
BANNER_HEIGHT = 45


class FireSmokeDetector:
    """Finds fire and smoke in a frame. Detections look like:
    {"kind": "FIRE" or "SMOKE", "conf": 0.0-1.0, "box": (x1, y1, x2, y2)}"""

    def __init__(self, model):
        self.model = model
        self.ids = {}                       # class number -> "FIRE" / "SMOKE"
        # Read the class names from the model itself instead of guessing the numbers
        for number, name in model.names.items():
            text = str(name).strip().lower()
            if "fire" in text or "flame" in text:
                self.ids[int(number)] = "FIRE"
            elif "smoke" in text:
                self.ids[int(number)] = "SMOKE"
        if not self.ids:
            raise ValueError("This model has no fire or smoke class.")

    @classmethod
    def load(cls):
        from ultralytics import YOLO
        if os.path.exists(LOCAL_MODEL):
            path = LOCAL_MODEL
        else:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(repo_id=FIRE_SMOKE_REPO, filename=FIRE_SMOKE_FILE)
        return cls(YOLO(path))

    def detect(self, frame, conf=0.45):
        results = self.model(frame, classes=list(self.ids), conf=conf, verbose=False)
        found = []
        for b in results[0].boxes:
            kind = self.ids.get(int(b.cls[0]))
            if kind is None:
                continue
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            found.append({"kind": kind, "conf": float(b.conf[0]), "box": (x1, y1, x2, y2)})
        return found


def draw_fire_smoke(frame, detections, banner_y=0):
    """Draws boxes and a banner. Returns the height of the banner (so the dashboard can stack text)."""
    if not detections:
        return 0
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        color = FIRE_COLOR if d["kind"] == "FIRE" else SMOKE_COLOR
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, f"{d['kind']} {int(d['conf'] * 100)}%",
                    (x1, max(y1 - 10, banner_y + BANNER_HEIGHT + 15)), FONT, 0.7, color, 2)

    fire = [d for d in detections if d["kind"] == "FIRE"]
    if fire:
        best = max(fire, key=lambda d: d["conf"])
        text, bg = f"CRITICAL: FIRE DETECTED {int(best['conf'] * 100)}%", (0, 0, 200)
    else:
        best = max(detections, key=lambda d: d["conf"])
        text, bg = f"WARNING: SMOKE DETECTED {int(best['conf'] * 100)}%", (90, 90, 90)
    w = frame.shape[1]
    cv2.rectangle(frame, (0, banner_y), (w, banner_y + BANNER_HEIGHT), bg, -1)
    cv2.putText(frame, text, (10, banner_y + 32), FONT, 0.8, (255, 255, 255), 2)
    return BANNER_HEIGHT


class FireSmokeAlerts:
    """Decides when an alert is due.
    - A kind (FIRE or SMOKE) must be seen in `confirm` checks in a row (stops one-frame false alarms).
    - While it stays in view, the alert repeats every `repeat_seconds`.
    - If it leaves the view for `gone_seconds`, the next appearance alerts again.
    - FIRE and SMOKE are tracked separately, so a smoke alert never hides a later fire alert."""

    KINDS = ("FIRE", "SMOKE")

    def __init__(self, confirm=2, repeat_seconds=120, gone_seconds=5):
        self.confirm = confirm
        self.repeat_seconds = repeat_seconds
        self.gone_seconds = gone_seconds
        self.hits = {k: 0 for k in self.KINDS}
        self.last_seen = {k: 0.0 for k in self.KINDS}
        self.last_alert = {k: 0.0 for k in self.KINDS}

    def configure(self, confirm, repeat_seconds):
        self.confirm = int(confirm)
        self.repeat_seconds = float(repeat_seconds)

    def update(self, detections, now):
        """Call after every fire/smoke check, also when nothing was found."""
        present = {d["kind"] for d in detections}
        for kind in self.KINDS:
            if kind in present:
                self.hits[kind] += 1
                self.last_seen[kind] = now
            else:
                self.hits[kind] = 0
                if self.last_seen[kind] and now - self.last_seen[kind] > self.gone_seconds:
                    self.last_alert[kind] = 0.0          # it left: a new appearance can alert again
                    self.last_seen[kind] = 0.0

    def due(self, detections, now):
        """Returns an alert (a dict) when one should be sent now, otherwise None."""
        present = {d["kind"] for d in detections}
        due_kinds = [k for k in self.KINDS
                     if k in present and self.hits[k] >= self.confirm
                     and now - self.last_alert[k] > self.repeat_seconds]
        if not due_kinds:
            return None
        for kind in due_kinds:
            self.last_alert[kind] = now

        has_fire = "FIRE" in due_kinds
        level = "CRITICAL" if has_fire else "WARNING"
        shown = sorted(detections, key=lambda d: (d["kind"] != "FIRE", -d["conf"]))
        detail = f"{level} | " + "; ".join(f"{d['kind']} {int(d['conf'] * 100)}%" for d in shown)
        return {
            "name": "fire" if has_fire else "smoke",      # used for the alert type and file names
            "level": level,
            "kinds": due_kinds,
            "detections": shown,
            "detail": detail,
            "photo_label": f"{level}: {' + '.join(sorted({d['kind'] for d in shown}))}",
        }


def build_email(alert, camera, when, clip_name, clip_before, clip_after):
    """Returns (subject, body) for the alert email."""
    fire = alert["name"] == "fire"
    what = "fire" if fire else "smoke"
    subject = f"{alert['level']}: Possible {what} detected - Digital Eye"
    lines = "\n".join(f"- {d['kind']} ({int(d['conf'] * 100)}% confidence)" for d in alert["detections"])
    body = (
        f"DIGITAL EYE - {what.upper()} ALERT ({alert['level']})\n\n"
        f"Possible {what} detected by the camera.\n"
        f"{lines}\n\n"
        f"Camera: {camera}\n"
        f"Time: {when}\n"
        "Photo is attached.\n"
        f"Video clip ({clip_before}s before, {clip_after}s after) is being saved: {clip_name} "
        "(dashboard > Recordings > Alert clips)\n\n"
        "This is an automatic AI detection and it can be wrong. Please check the photo. "
        "If you think there is a real fire, leave the area and call your local emergency service."
    )
    return subject, body