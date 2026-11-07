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
GRACE = 1.5            # seconds a person can vanish without resetting the zone timer
USE_FEET = False       # False = body center (best for webcam)

# Weapon model (downloaded once from Hugging Face, then cached on this computer)
WEAPON_REPO = "Subh775/Threat-Detection-YOLOv8n"
WEAPON_FILE = "weights/best.pt"
LOCAL_WEAPON_MODEL = os.path.join("models", "weapon_best.pt")   # optional manual copy
WEAPON_CLASSES = {"gun", "knife", "grenade"}

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
            image_path TEXT,
            alert_type TEXT DEFAULT 'zone',
            details TEXT DEFAULT ''
        )
        """)
        cols = [r[1] for r in db.execute("PRAGMA table_info(incidents)")]
        for name, ddl in [
            ("seconds_in_zone", "INTEGER DEFAULT 0"),
            ("alert_type", "TEXT DEFAULT 'zone'"),
            ("details", "TEXT DEFAULT ''"),
        ]:
            if name not in cols:
                db.execute(f"ALTER TABLE incidents ADD COLUMN {name} {ddl}")
        db.commit()


def add_incident(timestamp, people, seconds, path, alert_type="zone", details=""):
    with closing(sqlite3.connect(DB_PATH)) as db:
        db.execute(
            "INSERT INTO incidents "
            "(timestamp, camera, people_count, seconds_in_zone, image_path, alert_type, details) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (timestamp, CAMERA_NAME, people, seconds, path, alert_type, details),
        )
        db.commit()


def alerts_today(alert_type=None):
    today = datetime.now().strftime("%Y-%m-%d")
    query = "SELECT COUNT(*) FROM incidents WHERE timestamp LIKE ?"
    params = [today + "%"]
    if alert_type:
        query += " AND alert_type = ?"
        params.append(alert_type)
    with closing(sqlite3.connect(DB_PATH)) as db:
        return db.execute(query, params).fetchone()[0]


def recent_incidents(limit=6):
    with closing(sqlite3.connect(DB_PATH)) as db:
        return db.execute(
            "SELECT timestamp, seconds_in_zone, image_path, alert_type, details "
            "FROM incidents ORDER BY id DESC LIMIT ?", (limit,)
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


# ---------------------------------------------------------------- models + camera
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")


@st.cache_resource
def load_weapon_model():
    if os.path.exists(LOCAL_WEAPON_MODEL):
        path = LOCAL_WEAPON_MODEL
    else:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=WEAPON_REPO, filename=WEAPON_FILE)
    model = YOLO(path)
    # Read the class names from the model itself instead of guessing numbers
    ids = [i for i, n in model.names.items() if str(n).strip().lower() in WEAPON_CLASSES]
    if not ids:
        ids = [i for i, n in model.names.items() if str(n).strip().lower() != "explosive"]
    return model, ids


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
cooldown = st.sidebar.slider("Seconds between zone alerts", 10, 300, 60)

st.sidebar.subheader("Restricted zone")
zx = st.sidebar.slider("Left to right (%)", 0, 100, (50, 100))
zy = st.sidebar.slider("Top to bottom (%)", 0, 100, (0, 100))

st.sidebar.subheader("Weapon detection")
weapon_on = st.sidebar.toggle("Weapon detection ON", value=True)
weapon_conf = st.sidebar.slider("Weapon confidence", 0.30, 0.90, 0.50, 0.05)
weapon_every = st.sidebar.slider("Check every N frames", 1, 10, 3)
weapon_confirm = st.sidebar.slider("Checks in a row to confirm", 1, 5, 2)
weapon_cooldown = st.sidebar.slider("Seconds between weapon alerts", 10, 300, 60)

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

c1, c2, c3, c4, c5 = st.columns(5)
people_ph = c1.empty()
status_ph = c2.empty()
zone_ph = c3.empty()
alerts_ph = c4.empty()
weapon_ph = c5.empty()

video_ph = st.empty()

st.subheader("Recent alerts")
history_ph = st.empty()


def show_metrics(people, status, dwell):
    people_ph.metric("People now", people)
    status_ph.metric("Status", status)
    zone_ph.metric("Time in zone", dwell)


def show_alert_count():
    alerts_ph.metric("Alerts today", alerts_today())
    weapon_ph.metric("Weapon alerts today", alerts_today("weapon"))


def render_history():
    rows = recent_incidents()
    with history_ph.container():
        if not rows:
            st.info("No alerts yet.")
            return
        cols = st.columns(3)
        for i, (ts, secs, path, atype, details) in enumerate(rows):
            if atype == "weapon":
                caption = f"{ts}  |  WEAPON: {details}"
            else:
                caption = f"{ts}  |  {secs}s in zone"
            with cols[i % 3]:
                if path and os.path.exists(path):
                    st.image(path, caption=caption, width=300)
                else:
                    st.write(caption + " (photo missing)")


show_alert_count()
render_history()

# ---------------------------------------------------------------- camera off
if not run_camera:
    release_cam()
    show_metrics("-", "CAMERA OFF", "-")
    video_ph.info("The camera is off. Turn on 'Camera ON' in the sidebar.")
    st.stop()

# ---------------------------------------------------------------- load models
with st.spinner("Loading AI models (the weapon model downloads once, about 6 MB)..."):
    model = load_model()
    weapon_model, weapon_ids = None, []
    if weapon_on:
        try:
            weapon_model, weapon_ids = load_weapon_model()
        except Exception as e:
            st.sidebar.error(
                "Weapon model could not be loaded. Check your internet, or download "
                "best.pt manually and save it as models/weapon_best.pt. "
                f"Details: {e}"
            )

cam = get_cam()

st.session_state.setdefault("last_alert", 0.0)
st.session_state.setdefault("last_weapon_alert_RED", 0.0)
st.session_state.setdefault("last_weapon_alert_GREEN", 0.0)
zone_start = None
last_inside = 0.0
frame_no = 0
weapon_hits = 0
last_weapons = []       # (x1, y1, x2, y2, name, confidence)

# ---------------------------------------------------------------- live loop
while True:
    ok, frame = cam.read()
    if not ok:
        video_ph.error("Cannot read from the camera. Close other apps using it "
                       "(Zoom, Teams, the old digital_eye.py window), then refresh.")
        release_cam()
        st.stop()

    frame_no += 1
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

    # Detect weapons (on the clean frame, every N frames to keep the video smooth)
    if weapon_model is not None and frame_no % weapon_every == 0:
        wres = weapon_model(frame, classes=weapon_ids, conf=weapon_conf, verbose=False)
        last_weapons = []
        for b in wres[0].boxes:
            wx1, wy1, wx2, wy2 = map(int, b.xyxy[0])
            name = str(weapon_model.names[int(b.cls[0])])
            wcx, wcy = (wx1 + wx2) // 2, (wy1 + wy2) // 2
            loc = "RED" if (zx1 <= wcx <= zx2 and zy1 <= wcy <= zy2) else "GREEN"
            held = any(ax1 <= wcx <= ax2 and ay1 <= wcy <= ay2
                       for (ax1, ay1, ax2, ay2, _, _, _) in detections)
            last_weapons.append((wx1, wy1, wx2, wy2, name, float(b.conf[0]), loc, held))
        weapon_hits = weapon_hits + 1 if last_weapons else 0
    weapon_seen = len(last_weapons) > 0
    weapon_red = any(d[6] == "RED" for d in last_weapons)
    weapon_green = any(d[6] == "GREEN" for d in last_weapons)

    # Zone dwell timer
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

    # Draw weapons (red box = weapon in RED zone, orange box = weapon in GREEN zone)
    for (wx1, wy1, wx2, wy2, name, conf, loc, held) in last_weapons:
        wcolor = (0, 0, 255) if loc == "RED" else (0, 128, 255)
        cv2.rectangle(frame, (wx1, wy1), (wx2, wy2), wcolor, 3)
        cv2.putText(frame, f"WEAPON {name} {int(conf * 100)}% ({loc} ZONE)",
                    (wx1, max(wy1 - 10, 60)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, wcolor, 2)

    if weapon_seen:
        best = max(last_weapons, key=lambda d: d[5])
        if weapon_red and weapon_green:
            banner, bcolor = "CRITICAL: WEAPONS IN RED + GREEN ZONES", (0, 0, 255)
        elif weapon_red:
            banner, bcolor = f"CRITICAL: WEAPON IN RED ZONE - {best[4]} {int(best[5] * 100)}%", (0, 0, 255)
        else:
            banner, bcolor = f"WARNING: WEAPON IN GREEN ZONE - {best[4]} {int(best[5] * 100)}%", (0, 100, 230)
        cv2.rectangle(frame, (0, 0), (w, 45), bcolor, -1)
        cv2.putText(frame, banner, (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    now = datetime.now()
    cv2.putText(frame, now.strftime("%Y-%m-%d %H:%M:%S"), (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if intruder:
        zone_text_y = 75 if weapon_seen else 30
        cv2.putText(frame, f"ALERT  In zone: {int(dwell)}s / {dwell_limit}s", (10, zone_text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Update dashboard
    status = "WEAPON!" if weapon_seen else ("ALERT" if intruder else "SAFE")
    show_metrics(people, status, f"{int(dwell)}s" if intruder else "0s")
    video_ph.image(frame, channels="BGR", width=720)

    # Zone alert: person stayed long enough
    if dwell >= dwell_limit and t - st.session_state["last_alert"] > cooldown:
        st.session_state["last_alert"] = t
        stamp = now.strftime("%Y%m%d_%H%M%S")
        path = f"incidents/{stamp}.jpg"
        cv2.imwrite(path, frame)
        add_incident(now.strftime("%Y-%m-%d %H:%M:%S"), people, int(dwell), path, "zone", "")
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

    # Weapon protocol: alert the owner for a weapon in the RED zone and/or the GREEN zone
    if weapon_seen and weapon_hits >= weapon_confirm:
        red_list = [d for d in last_weapons if d[6] == "RED"]
        green_list = [d for d in last_weapons if d[6] == "GREEN"]
        due_red = bool(red_list) and t - st.session_state["last_weapon_alert_RED"] > weapon_cooldown
        due_green = bool(green_list) and t - st.session_state["last_weapon_alert_GREEN"] > weapon_cooldown

        if due_red or due_green:
            if red_list:
                st.session_state["last_weapon_alert_RED"] = t
            if green_list:
                st.session_state["last_weapon_alert_GREEN"] = t

            level = "CRITICAL" if red_list else "HIGH"
            if red_list and green_list:
                where = "RED and GREEN zones"
            elif red_list:
                where = "RED (restricted) zone"
            else:
                where = "GREEN (safe) zone"

            found = red_list + green_list
            lines = [
                f"- {d[4]} ({int(d[5] * 100)}% confidence) in the {d[6]} zone, "
                f"held by a person: {'YES' if d[7] else 'not clear'}"
                for d in found
            ]
            detail = f"{level} | " + "; ".join(f"{d[4]} {int(d[5] * 100)}% {d[6]}" for d in found)

            stamp = now.strftime("%Y%m%d_%H%M%S")
            path = f"incidents/weapon_{stamp}.jpg"
            cv2.imwrite(path, frame)
            add_incident(now.strftime("%Y-%m-%d %H:%M:%S"), people, int(dwell), path,
                         "weapon", detail)
            print(f"WEAPON incident saved: {path} ({detail})")

            body = (
                f"DIGITAL EYE - WEAPON ALERT ({level})\n\n"
                f"Possible weapon detected in the {where}.\n"
                + "\n".join(lines) + "\n\n"
                f"Person inside the restricted zone: {'YES' if intruder else 'No'}\n"
                f"People in view: {people}\n"
                f"Camera: {CAMERA_NAME}\n"
                f"Time: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
                "Photo is attached.\n\n"
                "This is an automatic AI detection and it can be wrong. "
                "Please check the photo before taking action."
            )
            threading.Thread(
                target=send_email,
                args=(f"{level}: Weapon detected in {where} - Digital Eye", body, path),
                daemon=True,
            ).start()

            show_alert_count()
            render_history()