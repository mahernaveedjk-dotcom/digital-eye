import collections
import glob
import hmac
import json
import os
import queue
import re
import shutil
import smtplib
import sqlite3
import threading
import time
from contextlib import closing
from datetime import datetime
from email.message import EmailMessage

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except ImportError:
    streamlit_image_coordinates = None

try:
    import imageio_ffmpeg          # makes MP4 files that play inside the dashboard
except ImportError:
    imageio_ffmpeg = None

from config import DASHBOARD_PASSWORD, EMAIL_FROM, EMAIL_APP_PASSWORD, EMAIL_TO

CAMERA_NAME = "Camera 1"
CAMERA_INDEX = 0
DB_PATH = "digital_eye.db"
ZONES_FILE = "zones.json"
RECORDINGS_DIR = "recordings"                       # manual recordings
CONTINUOUS_DIR = os.path.join(RECORDINGS_DIR, "continuous")   # continuous recording chunks
CLIPS_DIR = os.path.join(RECORDINGS_DIR, "clips")             # one clip for every alert
GRACE = 1.5               # seconds a person can vanish without resetting a zone timer
USE_FEET = False          # False = body center (best for webcam)
WARNING_SENDS_EMAIL = False   # True = "Warning" zones also send an email

# Smoothness and reliability settings
PERSON_EVERY = 2          # detect people on every 2nd camera frame (zone checks run on every frame)
DISPLAY_FPS = 15          # how often the picture is sent to the browser
RECONNECT_WAIT = 2.0      # seconds between camera reconnect attempts
WEAPON_GONE_SECONDS = 5   # weapon out of view this long = a new weapon alert is allowed
RECORD_FPS = 20           # frame rate of saved recordings
MAX_RECORD_SECONDS = 1800 # a manual recording stops by itself after 30 minutes
CHUNK_SECONDS = 300       # continuous recording is saved in 5-minute files
CLIP_BEFORE = 10          # alert clip: seconds BEFORE the alert
CLIP_AFTER = 20           # alert clip: seconds AFTER the alert
MAX_CLIP_SECONDS = 180    # an alert clip never gets longer than this
MIN_FREE_GB = 2           # continuous recording pauses when free disk space is below this
DETECTION_SIZES = {"Fast": 320, "Balanced": 416, "Accurate": 640}

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
os.makedirs(RECORDINGS_DIR, exist_ok=True)
os.makedirs(CONTINUOUS_DIR, exist_ok=True)
os.makedirs(CLIPS_DIR, exist_ok=True)

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


# ---------------------------------------------------------------- helpers
def fmt_time(seconds):
    seconds = int(seconds)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def put_label(img, text, org, color=(255, 255, 255), scale=0.7):
    """Text with a dark outline so it is readable on any background."""
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), 4)
    cv2.putText(img, text, org, FONT, scale, color, 2)


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
            details TEXT DEFAULT '',
            clip_path TEXT DEFAULT ''
        )
        """)
        cols = [r[1] for r in db.execute("PRAGMA table_info(incidents)")]
        for name, ddl in [
            ("seconds_in_zone", "INTEGER DEFAULT 0"),
            ("alert_type", "TEXT DEFAULT 'zone'"),
            ("details", "TEXT DEFAULT ''"),
            ("clip_path", "TEXT DEFAULT ''"),
        ]:
            if name not in cols:
                db.execute(f"ALTER TABLE incidents ADD COLUMN {name} {ddl}")
        db.commit()


def add_incident(timestamp, people, seconds, path, alert_type="zone", details="", clip_path=""):
    with closing(sqlite3.connect(DB_PATH)) as db:
        db.execute(
            "INSERT INTO incidents "
            "(timestamp, camera, people_count, seconds_in_zone, image_path, alert_type, details, clip_path) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (timestamp, CAMERA_NAME, people, seconds, path, alert_type, details, clip_path),
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
            "SELECT timestamp, seconds_in_zone, image_path, alert_type, details, clip_path "
            "FROM incidents ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()


init_db()


# ---------------------------------------------------------------- zones (saved in zones.json)
def save_zones(zones):
    with open(ZONES_FILE, "w", encoding="utf-8") as f:
        json.dump(zones, f, indent=2)


def _is_old_default(z):
    return (z.get("name") == "Zone 1" and z.get("type", "Restricted") == "Restricted"
            and abs(z.get("x1", -1) - 0.5) < 1e-6 and abs(z.get("y1", -1) - 0.0) < 1e-6
            and abs(z.get("x2", -1) - 1.0) < 1e-6 and abs(z.get("y2", -1) - 1.0) < 1e-6
            and z.get("timer", 10) == 10)


def load_zones():
    zones = None
    if os.path.exists(ZONES_FILE):
        try:
            with open(ZONES_FILE, encoding="utf-8") as f:
                zones = json.load(f)
        except Exception:
            zones = None
    if not isinstance(zones, list):
        return []          # no zones at all until you draw one
    # Older versions created a default zone on the right half of the screen. Remove it once.
    kept = [z for z in zones if not _is_old_default(z)]
    if len(kept) != len(zones):
        save_zones(kept)
        zones = kept
    for z in zones:
        z.setdefault("enabled", True)
        z.setdefault("timer", 10)
        z.setdefault("type", "Restricted")
    return zones


def zone_color(z):
    return RESTRICTED_COLOR if z["type"] == "Restricted" else WARNING_COLOR


def draw_zone_box(img, z, w, h, active=True):
    """active=True: outline in the zone's color (red/yellow). active=False: plain white outline."""
    x1, y1, x2, y2 = int(w * z["x1"]), int(h * z["y1"]), int(w * z["x2"]), int(h * z["y2"])
    if not z.get("enabled", True):
        color = (150, 150, 150)
    elif active:
        color = zone_color(z)
    else:
        color = (255, 255, 255)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
    cv2.putText(img, f"{z['name']} - {z['type']}", (x1 + 5, y1 + 18), FONT, 0.5, color, 1)


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


# ---- CameraReader start
class VideoFile:
    """One MP4 file. Frames are written at a steady frame rate, so playback speed is always right.
    A background thread does the encoding, so the camera never waits for the video encoder.
    Uses H.264 (plays in browsers) when imageio-ffmpeg is installed.
    live=True: the file stays readable even if the computer crashes while it is being written.
    block=True: never drop frames (used when a clip is saved from the buffer)."""

    def __init__(self, path, size, live=False, block=False):
        self.path = path
        self.size = size                   # (width, height)
        self.next_write = None
        self.block = block
        self.dropped = 0
        self.failed = False
        self.queue = queue.Queue(maxsize=200)
        if imageio_ffmpeg is not None:
            params = ["-preset", "veryfast"]
            if live:
                params += ["-movflags", "frag_keyframe+empty_moov+default_base_moof",
                           "-tune", "zerolatency", "-g", str(RECORD_FPS), "-flush_packets", "1"]
            self.kind = "ffmpeg"
            self.writer = imageio_ffmpeg.write_frames(
                path, size, pix_fmt_in="bgr24", fps=RECORD_FPS, codec="libx264",
                macro_block_size=2, quality=6, output_params=params)
            self.writer.send(None)
        else:
            self.kind = "cv2"
            self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), RECORD_FPS, size)
        self.thread = threading.Thread(target=self._worker, daemon=False)
        self.thread.start()

    def _worker(self):
        while True:
            frame = self.queue.get()
            if frame is None:
                break
            if self.failed:
                continue
            try:
                if (frame.shape[1], frame.shape[0]) != self.size:
                    frame = cv2.resize(frame, self.size)
                if self.kind == "ffmpeg":
                    self.writer.send(np.ascontiguousarray(frame).tobytes())
                else:
                    self.writer.write(frame)
            except Exception as e:
                print("Video encoder problem:", e)
                self.failed = True
        try:
            if self.kind == "ffmpeg":
                self.writer.close()
            else:
                self.writer.release()
        except Exception as e:
            print("Closing video file problem:", e)

    def _put(self, frame):
        if self.block:
            self.queue.put(frame)
        else:
            try:
                self.queue.put_nowait(frame)
            except queue.Full:
                self.dropped += 1          # the encoder cannot keep up: skip this frame

    def write_timed(self, frame, when):
        """Write the frame so that the video keeps real-time speed (repeats or skips frames if needed)."""
        if self.next_write is None:
            self.next_write = when
        count = 0
        while when >= self.next_write and count < 5:
            self._put(frame)
            self.next_write += 1.0 / RECORD_FPS
            count += 1
        if when - self.next_write > 1.0:
            self.next_write = when

    def close(self):
        self.queue.put(None)
        self.thread.join(timeout=60)


def _new_video_path(folder, prefix, suffix=""):
    name = datetime.now().strftime(f"{prefix}_%Y%m%d_%H%M%S") + suffix
    path = os.path.join(folder, name + ".mp4")
    n = 2
    while os.path.exists(path):
        path = os.path.join(folder, f"{name}_{n}.mp4")
        n += 1
    return path


class CameraReader:
    """Reads the camera in a background thread, so the dashboard always gets the newest
    frame (no lag). Reconnects by itself if the camera disconnects. It also records video
    in the background: continuous recording in chunks, a clip around every alert
    (kept from a rolling buffer, so it includes the seconds BEFORE the alert), and
    manual recordings. None of this stutters when the dashboard page reruns."""

    def __init__(self, index=CAMERA_INDEX):
        self.index = index
        self.lock = threading.Lock()       # protects the latest frame
        self.wlock = threading.Lock()      # protects all video writers and the clip state
        self.frame = None
        self.frame_id = 0
        self.running = False
        self.gen = 0                       # lets an old thread notice that it was replaced
        self.status = "stopped"            # stopped / connecting / live / reconnecting
        self.thread = None
        # manual recording
        self.rec_flag = False
        self.rec_started = 0.0
        self.rec_path = None
        self.last_recording = None
        self.rec_error = ""
        self._manual = None
        # continuous recording
        self.continuous = False
        self.cont_blocked = False
        self.cont_error = ""
        self._cont = None
        self._cont_started = 0.0
        self._disk_checked = 0.0
        # rolling buffer for alert clips: (time, JPEG bytes) at about RECORD_FPS
        self._prebuf = collections.deque(maxlen=int(CLIP_BEFORE * RECORD_FPS * 2) + 10)
        self._buf_next = 0.0
        self._clip = None
        self.finished_clips = []

    # ---- camera
    def start(self):
        if self.running and self.thread is not None and self.thread.is_alive():
            return
        self.running = True
        self.status = "connecting"
        self.gen += 1
        self.thread = threading.Thread(target=self._loop, args=(self.gen,), daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        with self.wlock:
            self.rec_flag = False
            self._close_manual()
            self._close_continuous()
            self._finish_clip()
        thread = self.thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=3)
        self.thread = None
        self.status = "stopped"
        with self.lock:
            self.frame = None

    def get_frame(self, last_id=-1):
        """Returns (frame_id, frame). frame is None if there is no newer frame."""
        with self.lock:
            if self.frame is None:
                return 0, None
            if self.frame_id == last_id:
                return self.frame_id, None
            return self.frame_id, self.frame.copy()

    def _loop(self, gen):
        first = True
        while self.running and self.gen == gen:
            cap = None
            try:
                self.status = "connecting" if first else "reconnecting"
                cap = cv2.VideoCapture(self.index)
                if not cap.isOpened():
                    raise RuntimeError("camera did not open")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                fails = 0
                while self.running and self.gen == gen:
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        fails = 0
                        first = False
                        self.status = "live"
                        with self.lock:
                            self.frame = frame
                            self.frame_id += 1
                        self._record(frame)
                    else:
                        fails += 1
                        with self.wlock:
                            self._check_clip(time.time())
                        if fails >= 30:
                            raise RuntimeError("camera stopped sending frames")
                        time.sleep(0.05)
            except Exception as e:
                print("Camera problem:", e)
            finally:
                if cap is not None:
                    cap.release()
            if self.running and self.gen == gen:
                self.status = "reconnecting"
                time.sleep(RECONNECT_WAIT)

    # ---- everything that is recorded, for each camera frame
    def _record(self, frame):
        now = time.time()
        with self.wlock:
            stamped = frame.copy()
            text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            org = (10, stamped.shape[0] - 12)
            cv2.putText(stamped, text, org, FONT, 0.6, (0, 0, 0), 3)
            cv2.putText(stamped, text, org, FONT, 0.6, (255, 255, 255), 1)
            try:
                self._write_manual(stamped, now)
                self._write_continuous(stamped, now)
                self._sample(stamped, now)
            except Exception as e:
                print("Recording problem:", e)
            self._check_clip(now)

    # ---- manual recording (Start / Stop button)
    @property
    def recording(self):
        return self.rec_flag

    def rec_elapsed(self):
        return (time.time() - self.rec_started) if self.rec_flag else 0.0

    def start_recording(self):
        with self.wlock:
            if self.rec_flag:
                return
            self.rec_error = ""
            self.rec_started = time.time()
            self.rec_flag = True

    def stop_recording(self):
        """Stops the manual recording and returns the saved file path (or None)."""
        with self.wlock:
            self.rec_flag = False
            return self._close_manual()

    def _write_manual(self, stamped, now):
        if not self.rec_flag:
            return
        if now - self.rec_started > MAX_RECORD_SECONDS:
            self.rec_flag = False
            self._close_manual()
            self.rec_error = "maximum recording length reached"
            return
        try:
            if self._manual is None:
                h, w = stamped.shape[:2]
                self.rec_path = _new_video_path(RECORDINGS_DIR, "rec")
                self._manual = VideoFile(self.rec_path, (w, h), live=True)
            self._manual.write_timed(stamped, now)
        except Exception as e:
            print("Manual recording problem:", e)
            self.rec_error = str(e)
            self.rec_flag = False
            self._close_manual()

    def _close_manual(self):
        path = None
        if self._manual is not None:
            self._manual.close()
            path = self.rec_path
            self.last_recording = path
        self._manual = None
        return path

    # ---- continuous recording (saved in chunks)
    def _write_continuous(self, stamped, now):
        if not self.continuous or self.cont_blocked:
            if self._cont is not None:
                self._close_continuous()
            return
        if now - self._disk_checked > 30:
            self._disk_checked = now
            try:
                free = shutil.disk_usage(CONTINUOUS_DIR).free
            except OSError:
                free = None
            if free is not None and free < MIN_FREE_GB * 1024 ** 3:
                self.cont_blocked = True
                self.cont_error = (f"Continuous recording paused: less than {MIN_FREE_GB} GB of disk "
                                   "space left. Delete old recordings, then switch it off and on again.")
                self._close_continuous()
                return
        if self._cont is not None and now - self._cont_started >= CHUNK_SECONDS:
            self._close_continuous()
        try:
            if self._cont is None:
                h, w = stamped.shape[:2]
                path = _new_video_path(CONTINUOUS_DIR, "cont")
                self._cont = VideoFile(path, (w, h), live=True)
                self._cont_started = now
            self._cont.write_timed(stamped, now)
        except Exception as e:
            print("Continuous recording problem:", e)
            self.cont_error = f"Continuous recording stopped: {e}"
            self.cont_blocked = True
            self._close_continuous()

    def _close_continuous(self):
        if self._cont is not None:
            self._cont.close()
        self._cont = None

    # ---- alert clips (10 s before + 20 s after)
    def _sample(self, stamped, now):
        if now < self._buf_next:
            return
        self._buf_next = now + 0.9 / RECORD_FPS
        ok, buf = cv2.imencode(".jpg", stamped, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return
        item = (now, buf.tobytes())
        self._prebuf.append(item)
        if self._clip is not None:
            self._clip["frames"].append(item)

    def request_clip(self, label):
        """Call this when an alert happens. Returns the path where the clip will appear
        (about CLIP_AFTER seconds later). Alerts that come close together share one clip."""
        with self.wlock:
            now = time.time()
            clip = self._clip
            if clip is not None:
                if now - clip["start"] < MAX_CLIP_SECONDS:
                    clip["deadline"] = now + CLIP_AFTER
                return clip["path"]
            safe = re.sub(r"[^A-Za-z0-9_-]+", "_", str(label))[:30]
            path = _new_video_path(CLIPS_DIR, "clip", "_" + safe)
            before = [it for it in self._prebuf if it[0] >= now - CLIP_BEFORE]
            self._clip = {"path": path, "start": now, "deadline": now + CLIP_AFTER,
                          "frames": before}
            return path

    def _check_clip(self, now):
        if self._clip is not None and now >= self._clip["deadline"]:
            self._finish_clip()

    def _finish_clip(self):
        clip, self._clip = self._clip, None
        if clip is not None and clip["frames"]:
            threading.Thread(target=self._write_clip, args=(clip,), daemon=False).start()

    def _write_clip(self, clip):
        try:
            frames = clip["frames"]
            first = cv2.imdecode(np.frombuffer(frames[0][1], np.uint8), cv2.IMREAD_COLOR)
            h, w = first.shape[:2]
            video = VideoFile(clip["path"], (w, h), block=True)
            for when, data in frames:
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    video.write_timed(img, when)
            video.close()
            with self.wlock:
                self.finished_clips.append(clip["path"])
        except Exception as e:
            print("Saving alert clip problem:", e)

    def pop_finished_clips(self):
        with self.wlock:
            done, self.finished_clips = self.finished_clips, []
        return done

    def active_paths(self):
        """Files that are being written right now."""
        with self.wlock:
            paths = set()
            if self._manual is not None:
                paths.add(os.path.abspath(self._manual.path))
            if self._cont is not None:
                paths.add(os.path.abspath(self._cont.path))
            return paths
# ---- CameraReader end


@st.cache_resource
def get_reader():
    """One camera reader for the whole app (shared, so a browser refresh never fights for the webcam)."""
    return CameraReader()


# ---------------------------------------------------------------- models
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


# ---------------------------------------------------------------- sidebar
reader = get_reader()

st.sidebar.title("Controls")
mode = st.sidebar.radio("Mode", ["Live", "Edit zones", "Recordings"])
run_camera = st.sidebar.toggle("Camera ON", value=True)
show_outlines = st.sidebar.toggle("Show zone outlines when empty", value=True)
repeat_minutes = st.sidebar.slider("Repeat zone alert every (minutes)", 1, 30, 5,
                                   help="While a person stays in a zone, the alert repeats this often")
repeat_seconds = repeat_minutes * 60
det_choice = st.sidebar.selectbox("Detection speed", list(DETECTION_SIZES), index=1,
                                  help="Fast = smoother video, Accurate = better detection")
det_size = DETECTION_SIZES[det_choice]

st.sidebar.subheader("Recording")
continuous_on = st.sidebar.toggle("Continuous recording (like CCTV)", value=True,
                                  help="Records all the time in 5-minute files. Nothing is deleted automatically.")
reader.continuous = continuous_on and run_camera
if not continuous_on:
    reader.cont_blocked = False
    reader.cont_error = ""
try:
    free_gb = shutil.disk_usage(RECORDINGS_DIR).free / 1024 ** 3
    st.sidebar.caption(f"Free disk space: {free_gb:.0f} GB")
except OSError:
    pass
if reader.cont_error:
    st.sidebar.error(reader.cont_error)
rec_label = "Stop recording" if reader.recording else "Start recording"
if st.sidebar.button(rec_label, disabled=not run_camera):
    if reader.recording:
        saved = reader.stop_recording()
        if saved:
            st.sidebar.success(f"Saved: {os.path.basename(saved)}")
        else:
            st.sidebar.warning("Nothing was saved (no camera frames yet).")
    else:
        reader.start()
        reader.start_recording()
rec_ph = st.sidebar.empty()
if reader.recording:
    rec_ph.error(f"● Recording {fmt_time(reader.rec_elapsed())}")
elif reader.rec_error:
    st.sidebar.warning(f"Recording stopped: {reader.rec_error}")
if imageio_ffmpeg is None:
    st.sidebar.caption("Install imageio-ffmpeg so recordings can play inside the dashboard.")

st.sidebar.subheader("Weapon detection")
weapon_on = st.sidebar.toggle("Weapon detection ON", value=True)
weapon_conf = st.sidebar.slider("Weapon confidence", 0.30, 0.90, 0.50, 0.05)
weapon_every = st.sidebar.slider("Check every N frames", 1, 10, 3)
weapon_confirm = st.sidebar.slider("Checks in a row to confirm", 1, 5, 2)
weapon_repeat_minutes = st.sidebar.slider("Repeat weapon alert every (minutes)", 1, 30, 2)
weapon_cooldown = weapon_repeat_minutes * 60

if st.sidebar.button("Send test email"):
    threading.Thread(
        target=send_email,
        args=("Digital Eye test", "This is a test email from the Digital Eye dashboard."),
        daemon=True,
    ).start()
    st.sidebar.success("Test email is being sent. Check your inbox.")

if st.sidebar.button("Log out"):
    reader.stop()
    st.session_state.auth = False
    st.rerun()


# ---------------------------------------------------------------- edit zones page
def take_snapshot():
    was_running = reader.running
    reader.start()
    frame = None
    deadline = time.time() + 10
    while time.time() < deadline:
        _, f = reader.get_frame()
        if f is not None:
            frame = f
            break
        time.sleep(0.1)
    if not was_running:
        reader.stop()
    return frame


def edit_zones_page():
    st.title("Edit zones")
    st.caption("Drag with the mouse on the picture to select an area. "
               "Then give it a name, a type, and a timer, and click Add zone.")

    if streamlit_image_coordinates is None:
        st.error("The drawing tool is not installed. Run this in the terminal, then refresh:\n\n"
                 "python -m pip install streamlit-image-coordinates")
        return

    if st.button("Take new snapshot") or "snapshot" not in st.session_state:
        snap = take_snapshot()
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


# ---------------------------------------------------------------- recordings page
def folder_size_mb(folder, recursive=False):
    pattern = os.path.join(folder, "**", "*.mp4") if recursive else os.path.join(folder, "*.mp4")
    return sum(os.path.getsize(f) for f in glob.glob(pattern, recursive=recursive)) / (1024 * 1024)


def video_browser(folder, key, bulk_delete=False):
    active = reader.active_paths()
    files = sorted((f for f in glob.glob(os.path.join(folder, "*.mp4"))
                    if os.path.abspath(f) not in active),
                   key=os.path.getmtime, reverse=True)
    if not files:
        st.info("Nothing here yet.")
        return

    def label(p):
        size_mb = os.path.getsize(p) / (1024 * 1024)
        when = datetime.fromtimestamp(os.path.getmtime(p)).strftime("%Y-%m-%d %H:%M:%S")
        return f"{os.path.basename(p)}   |   {when}   |   {size_mb:.1f} MB"

    choice = st.selectbox("Choose a video", files, format_func=label, key=f"{key}_pick")
    st.video(choice)

    c_dl, c_del = st.columns(2)
    if os.path.getsize(choice) < 100 * 1024 * 1024:
        with open(choice, "rb") as f:
            c_dl.download_button("Download", f.read(), file_name=os.path.basename(choice),
                                 mime="video/mp4", key=f"{key}_dl")
    else:
        c_dl.caption("File is large. Copy it from the folder: " + folder)
    if c_del.button("Delete this video", key=f"{key}_del"):
        try:
            os.remove(choice)
        except OSError as e:
            st.error(f"Could not delete: {e}")
        st.rerun()

    if bulk_delete:
        st.divider()
        days = st.number_input("Delete videos older than (days)", 1, 365, 7, key=f"{key}_days")
        sure = st.checkbox("I understand this cannot be undone", key=f"{key}_sure")
        if st.button("Delete old videos", key=f"{key}_bulk", disabled=not sure):
            limit = time.time() - days * 86400
            removed = 0
            for f in files:
                if os.path.getmtime(f) < limit:
                    try:
                        os.remove(f)
                        removed += 1
                    except OSError:
                        pass
            st.success(f"Deleted {removed} video(s).")
            st.rerun()


def recordings_page():
    st.title("Recordings")
    if imageio_ffmpeg is None:
        st.warning("Videos were saved in a format that browsers may not play. "
                   "Run: python -m pip install imageio-ffmpeg  (then record again).")
    try:
        free_gb = shutil.disk_usage(RECORDINGS_DIR).free / 1024 ** 3
    except OSError:
        free_gb = 0.0
    st.caption(
        f"Storage used: alert clips {folder_size_mb(CLIPS_DIR):.0f} MB  |  "
        f"continuous {folder_size_mb(CONTINUOUS_DIR):.0f} MB  |  "
        f"manual {folder_size_mb(RECORDINGS_DIR):.0f} MB  |  free disk {free_gb:.0f} GB")

    tab_clips, tab_cont, tab_manual = st.tabs(["Alert clips", "Continuous", "Manual"])
    with tab_clips:
        st.caption(f"One clip for each alert: {CLIP_BEFORE} seconds before and {CLIP_AFTER} seconds after.")
        video_browser(CLIPS_DIR, "clips")
    with tab_cont:
        st.caption(f"All-day recording, saved in {CHUNK_SECONDS // 60}-minute files. "
                   "The file being recorded right now appears here when it is finished.")
        video_browser(CONTINUOUS_DIR, "cont", bulk_delete=True)
    with tab_manual:
        st.caption("Recordings made with the Start recording button.")
        video_browser(RECORDINGS_DIR, "manual")


if not run_camera:
    reader.stop()
elif reader.continuous:
    reader.start()                # continuous recording keeps the camera on in every mode
elif mode != "Live" and not reader.recording:
    reader.stop()                 # free the webcam when it is not needed

if mode == "Edit zones":
    edit_zones_page()
    st.stop()

if mode == "Recordings":
    recordings_page()
    st.stop()

# ---------------------------------------------------------------- page layout (live)
st.title("Digital Eye - Live Security Dashboard")

c1, c2, c3, c4, c5, c6 = st.columns(6)
people_ph = c1.empty()
status_ph = c2.empty()
zone_ph = c3.empty()
alerts_ph = c4.empty()
weapon_ph = c5.empty()
cam_ph = c6.empty()

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
        for i, (ts, secs, path, atype, details, clip) in enumerate(rows):
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
                if clip:
                    if os.path.exists(clip):
                        with st.expander("Play video clip"):
                            st.video(clip)
                    else:
                        st.caption("Video clip is being saved...")


show_alert_count()
render_history()

# ---------------------------------------------------------------- camera off
if not run_camera:
    reader.stop()
    show_metrics("-", "CAMERA OFF", "-")
    cam_ph.metric("Camera", "OFF")
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

reader.start()

zones = [z for z in load_zones() if z.get("enabled", True)]
if not zones:
    st.info("No zones yet. This is a plain camera view. Choose 'Edit zones' in the sidebar to draw one.")
zstate = {z["id"]: {"start": None, "last_inside": 0.0} for z in zones}

st.session_state.setdefault("last_weapon_alert_RED", 0.0)
st.session_state.setdefault("last_weapon_alert_GREEN", 0.0)
frame_no = 0
weapon_hits = 0
person_boxes = []       # (x1, y1, x2, y2) from the latest person detection
last_weapons = []       # (x1, y1, x2, y2, name, confidence, RED/GREEN, held_by_person, zone_name)
last_id = -1
weapon_last_seen = 0.0
last_new = time.time()
last_display = 0.0
last_metrics = None
fps_count = 0
fps_t0 = time.time()

# ---------------------------------------------------------------- live loop
while True:
    fid, frame = reader.get_frame(last_id)
    t = time.time()

    if frame is None:
        # No new picture yet: either we are faster than the camera, or the camera is down
        if reader.status in ("connecting", "reconnecting") or t - last_new > 1.5:
            if reader.status == "connecting":
                video_ph.info("Starting the camera...")
            else:
                video_ph.warning("Camera disconnected. Reconnecting automatically...")
            show_metrics("-", "NO CAMERA", "-")
            cam_ph.metric("Camera", reader.status.upper())
            last_metrics = None
            time.sleep(0.25)
        else:
            time.sleep(0.005)
        continue

    last_id = fid
    last_new = t
    frame_no += 1
    fps_count += 1
    h, w = frame.shape[:2]

    zone_px = [(z, int(w * z["x1"]), int(h * z["y1"]), int(w * z["x2"]), int(h * z["y2"]))
               for z in zones]
    occupied = {z["id"]: False for z in zones}

    # Detect people (every PERSON_EVERY frames; zone checks below run on every frame)
    if frame_no == 1 or frame_no % PERSON_EVERY == 0:
        try:
            results = model(frame, classes=[0], imgsz=det_size, verbose=False)
            person_boxes = [tuple(map(int, b.xyxy[0])) for b in results[0].boxes]
        except Exception as e:
            print("Person detection problem:", e)
            person_boxes = []

    people = len(person_boxes)
    detections = []
    for (x1, y1, x2, y2) in person_boxes:
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
    if weapon_model is not None and (frame_no + 1) % weapon_every == 0:
        try:
            wres = weapon_model(frame, classes=weapon_ids, conf=weapon_conf, verbose=False)
            found_now = []
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
                found_now.append((wx1, wy1, wx2, wy2, name, float(b.conf[0]), loc, held, zname))
            last_weapons = found_now
            weapon_hits = weapon_hits + 1 if last_weapons else 0
        except Exception as e:
            print("Weapon detection problem:", e)
    weapon_seen = len(last_weapons) > 0
    if weapon_seen:
        weapon_last_seen = t
    elif weapon_last_seen and t - weapon_last_seen > WEAPON_GONE_SECONDS:
        st.session_state["last_weapon_alert_RED"] = 0.0
        st.session_state["last_weapon_alert_GREEN"] = 0.0
        weapon_last_seen = 0.0
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
            zs["start"] = None                                   # the visit is over
            st.session_state[f"last_alert_zone_{z['id']}"] = 0.0  # next visit can alert again

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

    # Zones are plain white outlines when empty (can be hidden) and turn red/yellow when occupied
    for z in zones:
        if occupied[z["id"]]:
            draw_zone_box(frame, z, w, h, active=True)
        elif show_outlines:
            draw_zone_box(frame, z, w, h, active=False)

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

    # Red REC mark on the live picture while recording
    if reader.recording:
        cv2.circle(frame, (w - 100, 22), 7, (0, 0, 255), -1)
        put_label(frame, f"REC {fmt_time(reader.rec_elapsed())}", (w - 88, 29), scale=0.55)

    # Update the dashboard (only when something changed, to keep it smooth)
    if weapon_seen:
        status = "WEAPON!"
    elif any_red:
        status = "ALERT"
    elif any_warn:
        status = "WARNING"
    else:
        status = "SAFE"
    n_occupied = sum(1 for v in occupied.values() if v)
    metrics = (people, status, f"{n_occupied} / {len(zones)}")
    if metrics != last_metrics:
        show_metrics(*metrics)
        last_metrics = metrics

    if t - last_display >= 1.0 / DISPLAY_FPS:
        last_display = t
        video_ph.image(frame, channels="BGR", output_format="JPEG", width=720)

    # Once per second: speed, camera status, recording timer
    if t - fps_t0 >= 1.0:
        fps = fps_count / (t - fps_t0)
        fps_count = 0
        fps_t0 = t
        cam_ph.metric("Camera", "LIVE", f"{fps:.0f} FPS", delta_color="off")
        if reader.pop_finished_clips():
            render_history()
        if reader.recording:
            rec_ph.error(f"● Recording {fmt_time(reader.rec_elapsed())}")
        else:
            rec_ph.empty()

    # Zone alerts: a person stayed long enough in a zone (each zone has its own timer)
    for (z, *_r) in zone_px:
        zid = z["id"]
        if not occupied[zid]:
            continue
        d = dwell_of(zid)
        key = f"last_alert_zone_{zid}"
        if d >= z["timer"] and t - st.session_state.get(key, 0.0) > repeat_seconds:
            st.session_state[key] = t
            is_restricted = z["type"] == "Restricted"
            stamp = now.strftime("%Y%m%d_%H%M%S")
            safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", z["name"])[:30]
            path = f"incidents/{stamp}_{safe_name}.jpg"

            snap = frame.copy()
            put_label(snap, f"ZONE: {z['name']} ({z['type']}) - {int(d)}s", (10, h - 45),
                      zone_color(z))
            cv2.imwrite(path, snap)
            clip_path = reader.request_clip(z["name"])
            add_incident(now.strftime("%Y-%m-%d %H:%M:%S"), people, int(d), path,
                         "zone" if is_restricted else "warning", z["name"], clip_path)
            print(f"Incident saved: {path} (zone: {z['name']})")

            if is_restricted or WARNING_SENDS_EMAIL:
                body = (
                    "DIGITAL EYE ALERT\n\n"
                    f"A person stayed {int(d)} seconds in the zone: {z['name']} ({z['type']}).\n"
                    f"Camera: {CAMERA_NAME}\n"
                    f"Time: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    "Photo is attached.\n"
                    f"Video clip ({CLIP_BEFORE}s before, {CLIP_AFTER}s after) is being saved: "
                    f"{os.path.basename(clip_path)} (dashboard > Recordings > Alert clips)"
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
            clip_path = reader.request_clip("weapon")
            add_incident(now.strftime("%Y-%m-%d %H:%M:%S"), people,
                         int(max([dwell_of(z["id"]) for z in zones], default=0)),
                         path, "weapon", detail, clip_path)
            print(f"WEAPON incident saved: {path} ({detail})")

            body = (
                f"DIGITAL EYE - WEAPON ALERT ({level})\n\n"
                f"Possible weapon detected in the {where}.\n"
                + "\n".join(lines) + "\n\n"
                f"Person inside a restricted zone: {'YES' if any_red else 'No'}\n"
                f"People in view: {people}\n"
                f"Camera: {CAMERA_NAME}\n"
                f"Time: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
                "Photo is attached.\n"
                f"Video clip ({CLIP_BEFORE}s before, {CLIP_AFTER}s after) is being saved: "
                f"{os.path.basename(clip_path)} (dashboard > Recordings > Alert clips)\n\n"
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