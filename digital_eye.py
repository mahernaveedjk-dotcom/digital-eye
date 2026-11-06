
import os
import smtplib
import sqlite3
import threading
import time
from datetime import datetime
from email.message import EmailMessage

import cv2
from ultralytics import YOLO

from config import EMAIL_FROM, EMAIL_APP_PASSWORD, EMAIL_TO


# ================= CONFIGURATION =================

CAMERA_NAME = "Camera 1"
DWELL_SECONDS = 10
COOLDOWN = 60
GRACE = 1.5
USE_FEET = False

DB_PATH = "digital_eye.db"
INCIDENT_DIR = "incidents"

os.makedirs(INCIDENT_DIR, exist_ok=True)


# ================= DATABASE =================

db = sqlite3.connect(DB_PATH)

db.execute("""
CREATE TABLE IF NOT EXISTS incidents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    camera TEXT,
    people_count INTEGER,
    seconds_in_zone INTEGER DEFAULT 0,
    image_path TEXT
)
""")

# Automatically update an older database.
# Existing incident records are preserved.

columns = {
    row[1]
    for row in db.execute("PRAGMA table_info(incidents)").fetchall()
}

required_columns = {
    "timestamp": "TEXT",
    "camera": "TEXT",
    "people_count": "INTEGER",
    "seconds_in_zone": "INTEGER DEFAULT 0",
    "image_path": "TEXT"
}

for column, definition in required_columns.items():
    if column not in columns:
        db.execute(
            f"ALTER TABLE incidents ADD COLUMN {column} {definition}"
        )
        print(f"Database updated: added {column}")

db.commit()

print("Database ready!")


# ================= EMAIL FUNCTION =================

def send_email(subject, body, image_path):

    for attempt in range(1, 4):

        try:
            msg = EmailMessage()

            msg["Subject"] = subject
            msg["From"] = EMAIL_FROM
            msg["To"] = EMAIL_TO

            msg.set_content(body)

            if os.path.exists(image_path):
                with open(image_path, "rb") as file:
                    msg.add_attachment(
                        file.read(),
                        maintype="image",
                        subtype="jpeg",
                        filename=os.path.basename(image_path)
                    )

            with smtplib.SMTP(
                "smtp.gmail.com", 587, timeout=30
            ) as server:

                server.starttls()
                server.login(EMAIL_FROM, EMAIL_APP_PASSWORD)
                server.send_message(msg)

            print("Email sent successfully!")
            return

        except Exception as error:
            print(f"Email attempt {attempt} failed: {error}")

            if attempt < 3:
                time.sleep(5)

    print("Email could not be sent.")
    print("Incident is still saved locally.")


# ================= LOAD YOLO MODEL =================

print("Loading YOLO model...")

model = YOLO("yolov8n.pt")

print("YOLO model loaded!")


# ================= CAMERA =================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    db.close()
    raise SystemExit


prev_time = time.time()

zone_start = None
last_inside = 0
last_alert = 0


# ================= MAIN DETECTION LOOP =================

try:

    while True:

        ok, frame = cap.read()

        if not ok:
            print("Camera frame could not be read.")
            break

        h, w = frame.shape[:2]
        mid = w // 2

        t = time.time()

        # -------- PERSON DETECTION --------

        results = model(frame, classes=[0], verbose=False)

        people = 0
        intruder = False
        detections = []

        for box in results[0].boxes:

            people += 1

            x1, y1, x2, y2 = map(
                int, box.xyxy[0].tolist()
            )

            px = (x1 + x2) // 2
            py = y2 if USE_FEET else (y1 + y2) // 2

            inside = px > mid

            if inside:
                intruder = True

            detections.append(
                (x1, y1, x2, y2, px, py, inside)
            )

        # -------- DWELL TIMER --------

        if intruder:

            last_inside = t

            if zone_start is None:
                zone_start = t

        elif zone_start is not None:

            if t - last_inside > GRACE:
                zone_start = None

        dwell = (
            t - zone_start
            if zone_start is not None
            else 0
        )

        # -------- RED RESTRICTED ZONE --------

        if intruder:

            overlay = frame.copy()

            cv2.rectangle(
                overlay,
                (mid, 0),
                (w, h),
                (0, 0, 255),
                -1
            )

            frame = cv2.addWeighted(
                overlay, 0.3, frame, 0.7, 0
            )

        cv2.line(
            frame,
            (mid, 0),
            (mid, h),
            (255, 255, 255),
            1
        )

        cv2.putText(
            frame,
            "Restricted area >",
            (mid + 10, h - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )

        # -------- DRAW DETECTED PEOPLE --------

        for (x1, y1, x2, y2, px, py, inside) in detections:

            color = (
                (0, 0, 255)
                if inside
                else (0, 255, 0)
            )

            label = (
                "INTRUDER"
                if inside
                else "Person"
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            cv2.circle(
                frame,
                (px, py),
                6,
                color,
                -1
            )

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        # -------- DISPLAY INFORMATION --------

        now = datetime.now()

        fps = 1 / max(time.time() - prev_time, 0.001)
        prev_time = time.time()

        cv2.putText(
            frame,
            f"People: {people}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            now.strftime("%Y-%m-%d %H:%M:%S"),
            (10, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        if intruder:

            cv2.putText(
                frame,
                "ALERT: RESTRICTED AREA",
                (10, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                3
            )

            cv2.putText(
                frame,
                f"In zone: {int(dwell)}s / {DWELL_SECONDS}s",
                (10, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        else:

            cv2.putText(
                frame,
                "STATUS: SAFE",
                (10, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        # -------- SAVE INCIDENT AND SEND EMAIL --------

        if dwell >= DWELL_SECONDS and t - last_alert > COOLDOWN:

            last_alert = t

            stamp = now.strftime("%Y%m%d_%H%M%S")

            path = os.path.join(
                INCIDENT_DIR,
                f"{stamp}.jpg"
            )

            saved = cv2.imwrite(path, frame)

            if saved:

                db.execute(
                    """
                    INSERT INTO incidents
                    (timestamp, camera, people_count,
                     seconds_in_zone, image_path)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        now.strftime("%Y-%m-%d %H:%M:%S"),
                        CAMERA_NAME,
                        people,
                        int(dwell),
                        path
                    )
                )

                db.commit()

                print(f"Incident saved: {path}")

                body = (
                    "DIGITAL EYE SECURITY ALERT\n\n"
                    f"A person stayed {int(dwell)} seconds "
                    "in the restricted area.\n"
                    f"Camera: {CAMERA_NAME}\n"
                    f"Time: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    "Photo is attached."
                )

                threading.Thread(
                    target=send_email,
                    args=(
                        "Digital Eye ALERT: Person in restricted area",
                        body,
                        path
                    ),
                    daemon=True
                ).start()

            else:
                print("Could not save incident image!")

        # -------- DISPLAY CAMERA --------

        cv2.imshow("Digital Eye", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


# ================= CLEANUP =================

finally:

    cap.release()
    cv2.destroyAllWindows()
    db.close()

    print("Digital Eye stopped.")
