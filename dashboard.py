import hmac
import json
import os
import re
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

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except ImportError:
    streamlit_image_coordinates = None

from config import DASHBOARD_PASSWORD, EMAIL_FROM, EMAIL_APP_PASSWORD, EMAIL_TO

CAMERA_NAME = "Camera 1"
DB_PATH = "digital_eye.db"
ZONES_FILE = "zones.json"
GRACE = 1.5               # seconds a person can vanish without resetting a zone timer
USE_FEET = False          # False = body center (best for webcam)
WARNING_SENDS_EMAIL = False   # True = "Warning" zones also send an email

RESTRICTED_COLOR = (0, 0, 255)      # red (BGR)
WARNING_COLOR = (0, 200, 255)       # yellow-orange (BGR)
FONT = cv2.FONT_HERSHEY_SIMPLEX

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


# ---------------------------------------------------------------- zones (saved in zones.json)
def save_zones(zones):
    with open(ZONES_FILE, "w", encoding="utf-8") as f:
        json.dump(zones, f, indent=2)


def load_zones():
    zones = None
    if os.path.exists(ZONES_FILE):
        try:
            with open(ZONES_FILE, encoding="utf-8") as f:
                zones = json.load(f)
        except Exception:
            zones = None
    if not isinstance(zones, list):
        # First run: one restricted zone on the right half, like the earlier versions
        zones = [{"id": 1, "name": "Zone 1", "type": "Restricted", "timer": 10,
                  "enabled": True, "x1": 0.5, "y1": 0.0, "x2": 1.0, "y2": 1.0}]
        save_zones(zones)
    for z in zones:
        z.setdefault("enabled", True)
        z.setdefault("timer", 10)
        z.setdefault("type", "Restricted")
    return zones


def zone_color(z):
    return RESTRICTED_COLOR if z["type"] == "Restricted" else WARNING_COLOR


def draw_zone_box(img, z, w, h):
    x1, y1, x2, y2 = int(w * z["x1"]), int(h * z["y1"]), int(w * z["x2"]), int(h * z["y2"])
    color = zone_color(z) if z.get("enabled", True) else (150, 150, 150)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
    cv2.putText(img, f"{z['name']} - {z['type']}", (x1 + 5, y1 + 18), FONT, 0.5, color, 1)


def put_label(img, text, org, color=(255, 255, 255), scale=0.7):
    """Text with a dark outline so it is readable on any background."""
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), 4)
    cv2.putText(img, text, org, FONT, scale, color, 2)


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


def grab_snapshot():
    cam = get_cam()
    frame = None
    for _ in range(8):          # a few reads so the camera can adjust its exposure
        ok, f = cam.read()
        if ok:
            frame = f
    release_cam()
    return frame


# ---------------------------------------------------------------- sidebar
st.sidebar.title("Controls")
mode = st.sidebar.radio("Mode", ["Live", "Edit zones"])
run_camera = st.sidebar.toggle("Camera ON", value=True)
cooldown = st.sidebar.slider("Seconds between alerts (per zone)", 10, 300, 60)

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


# ---------------------------------------------------------------- edit zones page
def edit_zones_page():
    st.title("Edit zones")
    st.caption("Drag with the mouse on the picture to select an area. "
               "Then give it a name, a type, and a timer, and click Add zone.")

    if streamlit_image_coordinates is None:
        st.error("The drawing tool is not installed. Run this in the terminal, then refresh:\n\n"
                 "python -m pip install streamlit-image-coordinates")
        return

    if st.button("Take new snapshot") or "snapshot" not in st.session_state:
        snap = grab_snapshot()
        if snap is None:
            st.error("Cannot read from the camera. Close other apps using it "
                     "(Zoom, Teams), then press 'Take new snapshot'.")
            return
        st.session_state["snapshot"] = snap
        st.session_state["pending"] = None

    snapshot = st.session_state["snapshot"]
    zones = load_zones()
    h, w = snapshot.shape[:2]

    base = snapshot.copy()
    for z in zones:
        draw_zone_box(base, z, w, h)
    rgb = cv2.cvtColor(base, cv2.COLOR_BGR2RGB)

    value = streamlit_image_coordinates(rgb, key="zone_draw", click_and_drag=True,
                                        width=720, cursor="crosshair")

    # A new drag finished -> remember the selected area (as fractions of the picture)
    if value and value.get("unix_time") != st.session_state.get("last_drag_time"):
        st.session_state["last_drag_time"] = value.get("unix_time")
        vw, vh = max(value["width"], 1), max(value["height"], 1)
        fx1, fx2 = sorted((value["x1"] / vw, value["x2"] / vw))
        fy1, fy2 = sorted((value["y1"] / vh, value["y2"] / vh))
        fx1, fx2 = max(0.0, fx1), min(1.0, fx2)
        fy1, fy2 = max(0.0, fy1), min(1.0, fy2)
        if fx2 - fx1 > 0.03 and fy2 - fy1 > 0.03:
            st.session_state["pending"] = (fx1, fy1, fx2, fy2)
        else:
            st.session_state["pending"] = None
            st.warning("That area is too small. Press and hold the mouse, drag, then release.")

    pending = st.session_state.get("pending")
    if pending:
        preview = snapshot.copy()
        cv2.rectangle(preview, (int(w * pending[0]), int(h * pending[1])),
                      (int(w * pending[2]), int(h * pending[3])), (0, 255, 255), 3)
        st.image(preview, channels="BGR", width=360, caption="Selected area (not saved yet)")
    else:
        st.info("No area selected yet. Drag on the picture above.")

    with st.form("add_zone", clear_on_submit=True):
        name = st.text_input("Zone name", value=f"Zone {len(zones) + 1}")
        ztype = st.selectbox("Type", ["Restricted", "Warning"])
        timer = st.number_input("Seconds before alert", 1, 120, 10)
        add = st.form_submit_button("Add zone")
    if add:
        if not pending:
            st.error("Select an area on the picture first.")
        elif not name.strip():
            st.error("Please type a zone name.")
        else:
            new_id = max([z["id"] for z in zones], default=0) + 1
            zones.append({"id": new_id, "name": name.strip(), "type": ztype,
                          "timer": int(timer), "enabled": True,
                          "x1": pending[0], "y1": pending[1],
                          "x2": pending[2], "y2": pending[3]})
            save_zones(zones)
            st.session_state["pending"] = None
            st.rerun()

    st.subheader("Saved zones")
    if not zones:
        st.info("No zones yet. People are detected, but no zone alerts will fire until you add one.")
    for z in zones:
        zid = z["id"]
        with st.expander(f"{z['name']}  |  {z['type']}  |  {z['timer']}s"
                         + ("" if z.get("enabled", True) else "  |  OFF")):
            n = st.text_input("Name", z["name"], key=f"n_{zid}")
            ty = st.selectbox("Type", ["Restricted", "Warning"],
                              index=0 if z["type"] == "Restricted" else 1, key=f"t_{zid}")
            tm = st.number_input("Seconds before alert", 1, 120, int(z["timer"]), key=f"s_{zid}")
            act = st.checkbox("Active", value=z.get("enabled", True), key=f"a_{zid}")
            cs, cd = st.columns(2)
            if cs.button("Save changes", key=f"sv_{zid}"):
                z.update({"name": n.strip() or z["name"], "type": ty,
                          "timer": int(tm), "enabled": act})
                save_zones(zones)
                st.rerun()
            if cd.button("Delete zone", key=f"dl_{zid}"):
                save_zones([q for q in zones if q["id"] != zid])
                st.rerun()


if mode == "Edit zones":
    edit_zones_page()
    st.stop()

# ---------------------------------------------------------------- page layout (live)
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


def show_metrics(people, status, zones_text):
    people_ph.metric("People now", people)
    status_ph.metric("Status", status)
    zone_ph.metric("Zones occupied", zones_text)


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
            elif atype == "warning":
                caption = f"{ts}  |  WARNING - {details}  |  {secs}s"
            else:
                caption = f"{ts}  |  {details or 'Zone'}  |  {secs}s in zone"
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

zones = [z for z in load_zones() if z.get("enabled", True)]
if not zones:
    st.warning("No active zones. Choose 'Edit zones' in the sidebar and draw one.")
zstate = {z["id"]: {"start": None, "last_inside": 0.0} for z in zones}

st.session_state.setdefault("last_weapon_alert_RED", 0.0)
st.session_state.setdefault("last_weapon_alert_GREEN", 0.0)
frame_no = 0
weapon_hits = 0
last_weapons = []       # (x1, y1, x2, y2, name, confidence, RED/GREEN, held_by_person, zone_name)

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
    t = time.time()

    zone_px = [(z, int(w * z["x1"]), int(h * z["y1"]), int(w * z["x2"]), int(h * z["y2"]))
               for z in zones]
    occupied = {z["id"]: False for z in zones}

    # Detect people and find which zones they stand in
    results = model(frame, classes=[0], verbose=False)
    people = 0
    detections = []

    for box in results[0].boxes:
        people += 1
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        px = (x1 + x2) // 2
        py = y2 if USE_FEET else (y1 + y2) // 2
        in_red = in_warn = False
        for (z, zx1, zy1, zx2, zy2) in zone_px:
            if zx1 <= px <= zx2 and zy1 <= py <= zy2:
                occupied[z["id"]] = True
                if z["type"] == "Restricted":
                    in_red = True
                else:
                    in_warn = True
        detections.append((x1, y1, x2, y2, px, py, in_red, in_warn))

    any_red = any(occupied[z["id"]] for z in zones if z["type"] == "Restricted")
    any_warn = any(occupied[z["id"]] for z in zones if z["type"] != "Restricted")

    # Detect weapons (on the clean frame, every N frames to keep the video smooth)
    if weapon_model is not None and frame_no % weapon_every == 0:
        wres = weapon_model(frame, classes=weapon_ids, conf=weapon_conf, verbose=False)
        last_weapons = []
        for b in wres[0].boxes:
            wx1, wy1, wx2, wy2 = map(int, b.xyxy[0])
            name = str(weapon_model.names[int(b.cls[0])])
            wcx, wcy = (wx1 + wx2) // 2, (wy1 + wy2) // 2

            # RED = inside a Restricted zone. GREEN = anywhere else.
            loc, zname = "GREEN", "outside all zones"
            for (z, zx1, zy1, zx2, zy2) in zone_px:
                if zx1 <= wcx <= zx2 and zy1 <= wcy <= zy2:
                    if z["type"] == "Restricted":
                        loc, zname = "RED", z["name"]
                        break
                    if zname == "outside all zones":
                        zname = z["name"]

            held = any(ax1 <= wcx <= ax2 and ay1 <= wcy <= ay2
                       for (ax1, ay1, ax2, ay2, *_rest) in detections)
            last_weapons.append((wx1, wy1, wx2, wy2, name, float(b.conf[0]), loc, held, zname))
        weapon_hits = weapon_hits + 1 if last_weapons else 0
    weapon_seen = len(last_weapons) > 0
    weapon_red = any(d[6] == "RED" for d in last_weapons)
    weapon_green = any(d[6] == "GREEN" for d in last_weapons)

    # Per-zone dwell timers
    for (z, *_r) in zone_px:
        zs = zstate[z["id"]]
        if occupied[z["id"]]:
            zs["last_inside"] = t
            if zs["start"] is None:
                zs["start"] = t
        elif zs["start"] is not None and t - zs["last_inside"] > GRACE:
            zs["start"] = None

    def dwell_of(zone_id):
        start = zstate[zone_id]["start"]
        return (t - start) if start else 0

    # Tint only the zones that someone is inside
    overlay = frame.copy()
    tinted = False
    for (z, zx1, zy1, zx2, zy2) in zone_px:
        if occupied[z["id"]]:
            cv2.rectangle(overlay, (zx1, zy1), (zx2, zy2), zone_color(z), -1)
            tinted = True
    if tinted:
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

    # Thin outline and name for every zone
    for z in zones:
        draw_zone_box(frame, z, w, h)

    # Draw people
    for (x1, y1, x2, y2, px, py, in_red, in_warn) in detections:
        if in_red:
            color, label = RESTRICTED_COLOR, "INTRUDER"
        elif in_warn:
            color, label = WARNING_COLOR, "WARNING"
        else:
            color, label = (0, 255, 0), "Person"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, (px, py), 6, color, -1)
        cv2.putText(frame, label, (x1, max(y1 - 10, 20)), FONT, 0.7, color, 2)

    # Draw weapons (red box = in a Restricted zone, orange box = anywhere else)
    for (wx1, wy1, wx2, wy2, name, conf, loc, held, zname) in last_weapons:
        wcolor = (0, 0, 255) if loc == "RED" else (0, 128, 255)
        cv2.rectangle(frame, (wx1, wy1), (wx2, wy2), wcolor, 3)
        cv2.putText(frame, f"WEAPON {name} {int(conf * 100)}% ({zname})",
                    (wx1, max(wy1 - 10, 60)), FONT, 0.55, wcolor, 2)

    if weapon_seen:
        best = max(last_weapons, key=lambda d: d[5])
        if weapon_red and weapon_green:
            banner, bcolor = "CRITICAL: WEAPONS IN RED + GREEN ZONES", (0, 0, 255)
        elif weapon_red:
            banner, bcolor = (f"CRITICAL: WEAPON IN RED ZONE - {best[4]} {int(best[5] * 100)}%",
                              (0, 0, 255))
        else:
            banner, bcolor = (f"WARNING: WEAPON IN GREEN ZONE - {best[4]} {int(best[5] * 100)}%",
                              (0, 100, 230))
        cv2.rectangle(frame, (0, 0), (w, 45), bcolor, -1)
        cv2.putText(frame, banner, (10, 32), FONT, 0.8, (255, 255, 255), 2)

    now = datetime.now()
    cv2.putText(frame, now.strftime("%Y-%m-%d %H:%M:%S"), (10, h - 15),
                FONT, 0.6, (255, 255, 255), 2)

    # Countdown line for every occupied zone
    text_y = 75 if weapon_seen else 30
    for (z, *_r) in zone_px:
        if occupied[z["id"]]:
            put_label(frame, f"{z['name']}: {int(dwell_of(z['id']))}s / {z['timer']}s",
                      (10, text_y), zone_color(z))
            text_y += 28

    # Update dashboard
    if weapon_seen:
        status = "WEAPON!"
    elif any_red:
        status = "ALERT"
    elif any_warn:
        status = "WARNING"
    else:
        status = "SAFE"
    n_occupied = sum(1 for v in occupied.values() if v)
    show_metrics(people, status, f"{n_occupied} / {len(zones)}")
    video_ph.image(frame, channels="BGR", width=720)

    # Zone alerts: a person stayed long enough in a zone (each zone has its own timer)
    for (z, *_r) in zone_px:
        zid = z["id"]
        if not occupied[zid]:
            continue
        d = dwell_of(zid)
        key = f"last_alert_zone_{zid}"
        if d >= z["timer"] and t - st.session_state.get(key, 0.0) > cooldown:
            st.session_state[key] = t
            is_restricted = z["type"] == "Restricted"
            stamp = now.strftime("%Y%m%d_%H%M%S")
            safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", z["name"])[:30]
            path = f"incidents/{stamp}_{safe_name}.jpg"

            snap = frame.copy()
            put_label(snap, f"ZONE: {z['name']} ({z['type']}) - {int(d)}s", (10, h - 45),
                      zone_color(z))
            cv2.imwrite(path, snap)
            add_incident(now.strftime("%Y-%m-%d %H:%M:%S"), people, int(d), path,
                         "zone" if is_restricted else "warning", z["name"])
            print(f"Incident saved: {path} (zone: {z['name']})")

            if is_restricted or WARNING_SENDS_EMAIL:
                body = (
                    "DIGITAL EYE ALERT\n\n"
                    f"A person stayed {int(d)} seconds in the zone: {z['name']} ({z['type']}).\n"
                    f"Camera: {CAMERA_NAME}\n"
                    f"Time: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    "Photo is attached."
                )
                threading.Thread(
                    target=send_email,
                    args=(f"Digital Eye ALERT: person in {z['name']}", body, path),
                    daemon=True,
                ).start()

            show_alert_count()
            render_history()

    # Weapon protocol: alert the owner for a weapon in a RED zone and/or the GREEN area
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
                where = "GREEN (safe) area"

            found = red_list + green_list
            zone_names = ", ".join(dict.fromkeys(d[8] for d in found))
            lines = [
                f"- {d[4]} ({int(d[5] * 100)}% confidence) in the {d[6]} zone - {d[8]}, "
                f"held by a person: {'YES' if d[7] else 'not clear'}"
                for d in found
            ]
            detail = f"{level} | " + "; ".join(
                f"{d[4]} {int(d[5] * 100)}% {d[6]} - {d[8]}" for d in found)

            stamp = now.strftime("%Y%m%d_%H%M%S")
            path = f"incidents/weapon_{stamp}.jpg"
            snap = frame.copy()
            put_label(snap, f"WEAPON in: {zone_names}", (10, h - 45), (0, 0, 255))
            cv2.imwrite(path, snap)
            add_incident(now.strftime("%Y-%m-%d %H:%M:%S"), people,
                         int(max([dwell_of(z["id"]) for z in zones], default=0)),
                         path, "weapon", detail)
            print(f"WEAPON incident saved: {path} ({detail})")

            body = (
                f"DIGITAL EYE - WEAPON ALERT ({level})\n\n"
                f"Possible weapon detected in the {where}.\n"
                + "\n".join(lines) + "\n\n"
                f"Person inside a restricted zone: {'YES' if any_red else 'No'}\n"
                f"People in view: {people}\n"
                f"Camera: {CAMERA_NAME}\n"
                f"Time: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
                "Photo is attached.\n\n"
                "This is an automatic AI detection and it can be wrong. "
                "Please check the photo before taking action."
            )
            threading.Thread(
                target=send_email,
                args=(f"{level}: Weapon detected - {zone_names} - Digital Eye", body, path),
                daemon=True,
            ).start()

            show_alert_count()
            render_history()