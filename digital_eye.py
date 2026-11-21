
import threading

import streamlit as st

from settings import DETECTION_SIZES
from database import init_db
from camera import CameraReader
from models import load_person_model
from email_alerts import send_email

from pages.login import show_login
from pages.zones_page import edit_zones_page
from pages.recordings_page import recordings_page

from live_dashboard import run_live_dashboard

from multi_camera import (
    MultiCameraManager,
    parse_camera_source,
    render_extra_cameras
)


# --------------------------------------------------
# Application setup
# --------------------------------------------------

st.set_page_config(
    page_title="Digital Eye",
    page_icon="👁️",
    layout="wide"
)

init_db()


# --------------------------------------------------
# Main camera
# --------------------------------------------------

@st.cache_resource
def get_reader():
    return CameraReader()


reader = get_reader()


# --------------------------------------------------
# Additional camera manager
# --------------------------------------------------

@st.cache_resource
def get_multi_camera_manager():
    return MultiCameraManager()


multi_manager = get_multi_camera_manager()


# --------------------------------------------------
# Login
# --------------------------------------------------

if not show_login():
    st.stop()


# --------------------------------------------------
# Sidebar controls
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
# Additional cameras
# --------------------------------------------------

st.sidebar.divider()
st.sidebar.subheader("Additional Cameras")

with st.sidebar.expander("Manage Cameras 2–4"):

    for number in range(2, 5):

        st.markdown(f"**Camera {number}**")

        source_text = st.text_input(
            f"Camera {number} source",
            value="",
            key=f"camera_source_{number}",
            placeholder="1 or RTSP URL",
            help="Enter a USB camera index or an RTSP stream URL."
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button(
                "Connect",
                key=f"connect_camera_{number}",
                use_container_width=True
            ):

                if not run_camera:
                    st.warning("Turn on Camera ON first.")

                elif not source_text.strip():
                    st.warning("Enter a camera source.")

                else:
                    source = parse_camera_source(source_text)

                    multi_manager.connect(
                        number,
                        source
                    )

                    st.success(
                        f"Camera {number} connection started."
                    )

        with col2:
            if st.button(
                "Disconnect",
                key=f"disconnect_camera_{number}",
                use_container_width=True
            ):

                multi_manager.disconnect(number)

                st.success(
                    f"Camera {number} disconnected."
                )

        camera = multi_manager.cameras.get(number)

        if camera:
            st.caption(
                f"Status: {camera.status.upper()}"
            )

        st.divider()


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
    st.sidebar.error("● Recording is active")


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
    multi_manager.stop_all()

    st.session_state.auth = False

    st.rerun()


# --------------------------------------------------
# Camera management
# --------------------------------------------------

if not run_camera:

    reader.stop()
    multi_manager.stop_all()

elif reader.continuous:

    reader.start()

elif mode != "Live" and not reader.recording:

    reader.stop()


# --------------------------------------------------
# Other pages
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
        "The cameras are off. Turn on 'Camera ON' in the sidebar."
    )

    st.metric("Main Camera", "OFF")

    st.stop()


# --------------------------------------------------
# Load person detection model
# --------------------------------------------------

with st.spinner("Loading person detection model..."):

    person_model = load_person_model()


# --------------------------------------------------
# Main live dashboard
# --------------------------------------------------

# Additional camera feeds
st.subheader("Additional Camera Feeds")
render_extra_cameras(multi_manager)

st.divider()

# Main camera dashboard
run_live_dashboard(
    reader=reader,
    model=person_model,
    repeat_seconds=repeat_seconds,
    detection_size=detection_size,
    show_outlines=show_outlines
)
