import hmac
import os
import smtplib
import sqlite3
import threading
import time
from contextlib import closing
from datetime import datetime
from email.message import EmailMessage

import cv2
import streamlit as st
from ultralytics import YOLO

from config import DASHBOARD_PASSWORD, EMAIL_FROM, EMAIL_APP_PASSWORD, EMAIL_TO

CAMERA_NAME = "Camera 1"
DB_PATH = "digital_eye.db"
GRACE = 1.5            # seconds a person can vanish without resetting the timer
USE_FEET = False       # False = body center (best for webcam)

st.set_page_config(page_title="Digital Eye", page_icon="👁️", layout="wide")
os.makedirs("incidents", exist_ok=True)

# ---------------------------------------------------------------- login
if "auth" not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    st.title("Digital Eye")
    st.caption("Security dashboard - please log in")
    with st.form("login"):
        pw = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")
    if submitted:
        if hmac.compare_digest(pw.encode(), DASHBOARD_PASSWORD.encode()):
            st.session_state.auth = True
            st.rerun()
        else:
            st.error("Wrong password")
    st.stop()


# ---------------------------------------------------------------- database
def init_db():
    with closing(sqlite3.connect(DB_PATH)) as db:
        db.execute("""
        CREATE TABLE IF NOT EXISTS incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            camera TEXT,
            people_count INTEGER,
            seconds_in_zone INTEGER,
            image_path TEXT
        )
        """)
        cols = [r[1] for r in db.execute("PRAGMA table_info(incidents)")]
        if "seconds_in_zone" not in cols:
            db.execute("ALTER TABLE incidents ADD COLUMN seconds_in_zone INTEGER DEFAULT 0")
        db.commit()


def add_incident(timestamp, people, seconds, path):
    with closing(sqlite3.connect(DB_PATH)) as db:
        db.execute(
            "INSERT INTO incidents (timestamp, camera, people_count, seconds_in_zone, image_path) "
            "VALUES (?, ?, ?, ?, ?)",
            (timestamp, CAMERA_NAME, people, seconds, path),
        )
        db.commit()


def alerts_today():
    today = datetime.now().strftime("%Y-%m-%d")
    with closing(sqlite3.connect(DB_PATH)) as db:
        return db.execute(
            "SELECT COUNT(*) FROM incidents WHERE timestamp LIKE ?", (today + "%",)
        ).fetchone()[0]


def recent_incidents(limit=6):
    with closing(sqlite3.connect(DB_PATH)) as db:
        return db.execute(
            "SELECT timestamp, seconds_in_zone, image_path FROM incidents "
            "ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()


init_db()


# ---------------------------------------------------------------- email
def send_email(subject, body, image_path=None):
    """Runs in the background so the dashboard never freezes. Tries 3 times."""
    for attempt in range(1, 4):
        try:
            msg = EmailMessage()
            msg["Subject"] = subject
            msg["From"] = EMAIL_FROM
            msg["To"] = EMAIL_TO
            msg.set_content(body)
            if image_path:
                with open(image_path, "rb") as f:
                    msg.add_attachment(f.read(), maintype="image", subtype="jpeg",
                                       filename=os.path.basename(image_path))
            with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
                server.starttls()
                server.login(EMAIL_FROM, EMAIL_APP_PASSWORD)
                server.send_message(msg)
            print("Email sent")
            return
        except Exception as e:
            print(f"Email attempt {attempt} failed:", e)
            time.sleep(5)
    print("Email could not be sent. The incident is still saved on this computer.")


# ---------------------------------------------------------------- camera + model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")


def get_cam():
    cam = st.session_state.get("cam")
    if cam is None or not cam.isOpened():
        cam = cv2.VideoCapture(0)
        st.session_state["cam"] = cam
    return cam


def release_cam():
    cam = st.session_state.pop("cam", None)
    if cam is not None:
        cam.release()


# ---------------------------------------------------------------- sidebar
st.sidebar.title("Controls")
run_camera = st.sidebar.toggle("Camera ON", value=True)
dwell_limit = st.sidebar.slider("Seconds in zone before alert", 3, 60, 10)
cooldown = st.sidebar.slider("Seconds between alerts", 10, 300, 60)

st.sidebar.subheader("Restricted zone")
zx = st.sidebar.slider("Left to right (%)", 0, 100, (50, 100))
zy = st.sidebar.slider("Top to bottom (%)", 0, 100, (0, 100))

if st.sidebar.button("Send test email"):
    threading.Thread(
        target=send_email,
        args=("Digital Eye test", "This is a test email from the Digital Eye dashboard."),
        daemon=True,
    ).start()
    st.sidebar.success("Test email is being sent. Check your inbox.")

if st.sidebar.button("Log out"):
    release_cam()
    st.session_state.auth = False
    st.rerun()

# ---------------------------------------------------------------- page layout
st.title("Digital Eye - Live Security Dashboard")

c1, c2, c3, c4 = st.columns(4)
people_ph = c1.empty()
status_ph = c2.empty()
zone_ph = c3.empty()
alerts_ph = c4.empty()

video_ph = st.empty()

st.subheader("Recent alerts")
history_ph = st.empty()


def show_metrics(people, status, dwell):
    people_ph.metric("People now", people)
    status_ph.metric("Status", status)
    zone_ph.metric("Time in zone", dwell)


def show_alert_count():
    alerts_ph.metric("Alerts today", alerts_today())


def render_history():
    rows = recent_incidents()
    with history_ph.container():
        if not rows:
            st.info("No alerts yet.")
            return
        cols = st.columns(3)
        for i, (ts, secs, path) in enumerate(rows):
            with cols[i % 3]:
                if path and os.path.exists(path):
                    st.image(path, caption=f"{ts}  |  {secs}s in zone", width=300)
                else:
                    st.write(f"{ts}  |  {secs}s in zone (photo missing)")


show_alert_count()
render_history()

# ---------------------------------------------------------------- camera off
if not run_camera:
    release_cam()
    show_metrics("-", "CAMERA OFF", "-")
    video_ph.info("The camera is off. Turn on 'Camera ON' in the sidebar.")
    st.stop()

# ---------------------------------------------------------------- live loop
with st.spinner("Loading AI model..."):
    model = load_model()
cam = get_cam()

st.session_state.setdefault("last_alert", 0.0)
zone_start = None
last_inside = 0.0

while True:
    ok, frame = cam.read()
    if not ok:
        video_ph.error("Cannot read from the camera. Close other apps using it "
                       "(Zoom, Teams, the old digital_eye.py window), then refresh.")
        release_cam()
        st.stop()

    h, w = frame.shape[:2]
    zx1, zx2 = int(w * zx[0] / 100), int(w * zx[1] / 100)
    zy1, zy2 = int(h * zy[0] / 100), int(h * zy[1] / 100)
    t = time.time()

    # Detect people
    results = model(frame, classes=[0], verbose=False)
    people = 0
    intruder = False
    detections = []

    for box in results[0].boxes:
        people += 1
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        px = (x1 + x2) // 2
        py = y2 if USE_FEET else (y1 + y2) // 2
        inside = zx1 <= px <= zx2 and zy1 <= py <= zy2
        if inside:
            intruder = True
        detections.append((x1, y1, x2, y2, px, py, inside))

    # Dwell timer
    if intruder:
        last_inside = t
        if zone_start is None:
            zone_start = t
    elif zone_start is not None and t - last_inside > GRACE:
        zone_start = None

    dwell = (t - zone_start) if zone_start else 0

    # Red tint only while someone is inside the zone
    if intruder:
        overlay = frame.copy()
        cv2.rectangle(overlay, (zx1, zy1), (zx2, zy2), (0, 0, 255), -1)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
    cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (255, 255, 255), 1)

    # Draw people
    for (x1, y1, x2, y2, px, py, inside) in detections:
        color = (0, 0, 255) if inside else (0, 255, 0)
        label = "INTRUDER" if inside else "Person"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, (px, py), 6, color, -1)
        cv2.putText(frame, label, (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    now = datetime.now()
    cv2.putText(frame, now.strftime("%Y-%m-%d %H:%M:%S"), (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if intruder:
        cv2.putText(frame, f"ALERT  In zone: {int(dwell)}s / {dwell_limit}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Update dashboard
    show_metrics(people, "ALERT" if intruder else "SAFE",
                 f"{int(dwell)}s" if intruder else "0s")
    video_ph.image(frame, channels="BGR", width=720)

    # Alert after the person stays long enough
    if dwell >= dwell_limit and t - st.session_state["last_alert"] > cooldown:
        st.session_state["last_alert"] = t
        stamp = now.strftime("%Y%m%d_%H%M%S")
        path = f"incidents/{stamp}.jpg"
        cv2.imwrite(path, frame)
        add_incident(now.strftime("%Y-%m-%d %H:%M:%S"), people, int(dwell), path)
        print(f"Incident saved: {path}")

        body = (
            "DIGITAL EYE ALERT\n\n"
            f"A person stayed {int(dwell)} seconds in the restricted area.\n"
            f"Camera: {CAMERA_NAME}\n"
            f"Time: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
            "Photo is attached."
        )
        threading.Thread(
            target=send_email,
            args=("Digital Eye ALERT: person in restricted area", body, path),
            daemon=True,
        ).start()

        show_alert_count()
        render_history()