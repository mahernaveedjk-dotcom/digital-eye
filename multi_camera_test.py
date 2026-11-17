
import cv2
import streamlit as st
import threading
import time

st.set_page_config(
    page_title="Digital Eye - Multiple Cameras",
    layout="wide"
)

st.title("Digital Eye - Multiple Camera System")

MAX_CAMERAS = 4


class Camera:
    def __init__(self, source, name):
        self.source = source
        self.name = name
        self.frame = None
        self.running = False
        self.status = "stopped"
        self.error = ""
        self.lock = threading.Lock()
        self.thread = None

    def start(self):
        if self.running:
            return

        self.running = True
        self.status = "connecting"
        self.error = ""

        self.thread = threading.Thread(
            target=self.capture_loop,
            daemon=True
        )
        self.thread.start()

    def capture_loop(self):
        while self.running:
            cap = None

            try:
                # Use DirectShow for Windows webcam indexes.
                if isinstance(self.source, int):
                    cap = cv2.VideoCapture(
                        self.source,
                        cv2.CAP_DSHOW
                    )
                else:
                    cap = cv2.VideoCapture(self.source)

                if not cap.isOpened():
                    raise RuntimeError(
                        f"Could not open camera source: {self.source}"
                    )

                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                while self.running:
                    success, frame = cap.read()

                    if not success or frame is None:
                        raise RuntimeError(
                            "Camera opened but no frame received"
                        )

                    rgb_frame = cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB
                    )

                    with self.lock:
                        self.frame = rgb_frame

                    self.status = "live"
                    self.error = ""

            except Exception as error:
                self.error = str(error)
                self.status = "reconnecting"
                print(f"{self.name} problem: {error}")

            finally:
                if cap is not None:
                    cap.release()

            if self.running:
                time.sleep(2)

        self.status = "stopped"

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def stop(self):
        self.running = False

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3)

        self.status = "stopped"

        with self.lock:
            self.frame = None


if "cameras" not in st.session_state:
    st.session_state.cameras = {}

st.sidebar.header("Camera Settings")

camera_number = st.sidebar.selectbox(
    "Select camera",
    [1, 2, 3, 4]
)

camera_name = f"Camera {camera_number}"

source_text = st.sidebar.text_input(
    "Camera source",
    value="0",
    help="Use 0 for laptop webcam. Use 1 for another webcam."
)

if st.sidebar.button("Connect Camera"):
    try:
        source_text = source_text.strip()

        if not source_text:
            raise ValueError("Enter a camera source.")

        source = (
            int(source_text)
            if source_text.isdigit()
            else source_text
        )

        old_camera = st.session_state.cameras.get(camera_name)

        if old_camera:
            old_camera.stop()

        camera = Camera(source, camera_name)
        st.session_state.cameras[camera_name] = camera
        camera.start()

        st.sidebar.success(
            f"{camera_name}: connection started"
        )

    except Exception as error:
        st.sidebar.error(str(error))


if st.sidebar.button("Stop Selected Camera"):
    camera = st.session_state.cameras.get(camera_name)

    if camera:
        camera.stop()
        st.sidebar.success(f"{camera_name} stopped")


@st.fragment(run_every=0.5)
def show_camera_grid():
    st.subheader("Live Camera Grid")

    columns = st.columns(2)

    for index in range(MAX_CAMERAS):
        name = f"Camera {index + 1}"

        with columns[index % 2]:
            st.markdown(f"### {name}")

            camera = st.session_state.cameras.get(name)

            if camera is None:
                st.info("Camera disconnected")
                continue

            frame = camera.get_frame()

            if frame is not None:
                st.image(frame, width="stretch")
                st.success("Connected")
            elif camera.status == "connecting":
                st.info("Connecting...")
            elif camera.status == "reconnecting":
                st.warning(
                    f"Connection failed: {camera.error}"
                )
            else:
                st.info(f"Status: {camera.status}")


show_camera_grid()

st.caption("Digital Eye - Multiple Camera Testing")
