import threading

import streamlit as st

from settings import (
    DETECTION_SIZES,
    RECORDINGS_DIR
)

from database import init_db
from camera import CameraReader
from models import load_person_model, load_weapon_model
from email_alerts import send_email

from pages.login import show_login
from pages.zones_page import edit_zones_page
from pages.recordings_page import recordings_page

from live_dashboard import run_live_dashboard


# --------------------------------------------------
# Application setup
# --------------------------------------------------

st.set_page_config(
    page_title="Digital Eye",
    page_icon="👁️",
    layout="wide"
)

init_db()


@st.cache_resource
def get_reader():
    return CameraReader()


reader = get_reader()


# --------------------------------------------------
# Login
# --------------------------------------------------

if not show_login():
    st.stop()


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("Controls")

mode = st.sidebar.radio(
    "Mode",
    ["Live", "Edit zones", "Recordings"]
)

run_camera = st.sidebar.toggle(
    "Camera ON",
    value=True
)

show_outlines = st.sidebar.toggle(
    "Show zone outlines when empty",
    value=True
)

repeat_minutes = st.sidebar.slider(
    "Repeat zone alert every (minutes)",
    1,
    30,
    5
)

repeat_seconds = repeat_minutes * 60

det_choice = st.sidebar.selectbox(
    "Detection speed",
    list(DETECTION_SIZES),
    index=1
)

detection_size = DETECTION_SIZES[det_choice]


# --------------------------------------------------
# Recording controls
# --------------------------------------------------

st.sidebar.subheader("Recording")

continuous_on = st.sidebar.toggle(
    "Continuous recording (like CCTV)",
    value=True
)

reader.continuous = continuous_on and run_camera

if not continuous_on:
    reader.cont_blocked = False
    reader.cont_error = ""

if reader.cont_error:
    st.sidebar.error(reader.cont_error)

recording_button = (
    "Stop recording"
    if reader.recording
    else "Start recording"
)

if st.sidebar.button(
    recording_button,
    disabled=not run_camera
):
    if reader.recording:
        saved = reader.stop_recording()

        if saved:
            st.sidebar.success(
                f"Saved: {saved.split(chr(92))[-1]}"
            )
        else:
            st.sidebar.warning(
                "Nothing was saved yet."
            )

    else:
        reader.start()
        reader.start_recording()

if reader.recording:
    st.sidebar.error(
        "● Recording is active"
    )

if reader.rec_error:
    st.sidebar.warning(
        f"Recording stopped: {reader.rec_error}"
    )

try:
    import imageio_ffmpeg

    st.sidebar.caption(
        "MP4 recording support is available."
    )

except ImportError:
    st.sidebar.caption(
        "Install imageio-ffmpeg for browser-compatible recordings."
    )


# --------------------------------------------------
# Weapon detection controls
# --------------------------------------------------

st.sidebar.subheader("Weapon detection")

weapon_on = st.sidebar.toggle(
    "Weapon detection ON",
    value=True
)

weapon_conf = st.sidebar.slider(
    "Weapon confidence",
    0.30,
    0.90,
    0.50,
    0.05
)

weapon_every = st.sidebar.slider(
    "Check every N frames",
    1,
    10,
    3
)

weapon_confirm = st.sidebar.slider(
    "Checks in a row to confirm",
    1,
    5,
    2
)

weapon_repeat_minutes = st.sidebar.slider(
    "Repeat weapon alert every (minutes)",
    1,
    30,
    2
)

weapon_cooldown = weapon_repeat_minutes * 60


# --------------------------------------------------
# Test email
# --------------------------------------------------

if st.sidebar.button("Send test email"):
    threading.Thread(
        target=send_email,
        args=(
            "Digital Eye test",
            "This is a test email from Digital Eye."
        ),
        daemon=True
    ).start()

    st.sidebar.success(
        "Test email is being sent. Check your inbox."
    )


# --------------------------------------------------
# Logout
# --------------------------------------------------

if st.sidebar.button("Log out"):
    reader.stop()

    st.session_state.auth = False

    st.rerun()


# --------------------------------------------------
# Camera management
# --------------------------------------------------

if not run_camera:
    reader.stop()

elif reader.continuous:
    reader.start()

elif mode != "Live" and not reader.recording:
    reader.stop()


# --------------------------------------------------
# Pages
# --------------------------------------------------

if mode == "Edit zones":
    edit_zones_page(reader)
    st.stop()


if mode == "Recordings":
    recordings_page(reader)
    st.stop()


# --------------------------------------------------
# Camera OFF screen
# --------------------------------------------------

if not run_camera:
    st.title("Digital Eye - Live Security Dashboard")

    st.info(
        "The camera is off. Turn on 'Camera ON' in the sidebar."
    )

    st.metric("Camera", "OFF")

    st.stop()


# --------------------------------------------------
# Load AI models
# --------------------------------------------------

with st.spinner("Loading AI models..."):
    person_model = load_person_model()

    weapon_model = None
    weapon_ids = []

    if weapon_on:
        try:
            weapon_model, weapon_ids = load_weapon_model()

        except Exception as error:
            st.sidebar.error(
                "Weapon model could not be loaded. "
                "Check your internet connection or place "
                "best.pt inside the models folder. "
                f"Details: {error}"
            )


# --------------------------------------------------
# Start live dashboard
# --------------------------------------------------

run_live_dashboard(
    reader=reader,
    model=person_model,
    weapon_model=weapon_model,
    weapon_ids=weapon_ids,
    weapon_on=weapon_on,
    weapon_conf=weapon_conf,
    weapon_every=weapon_every,
    weapon_confirm=weapon_confirm,
    weapon_cooldown=weapon_cooldown,
    repeat_seconds=repeat_seconds,
    detection_size=detection_size,
    show_outlines=show_outlines
)