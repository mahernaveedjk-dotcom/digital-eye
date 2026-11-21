
import cv2
import streamlit as st
import threading
import time


class CameraStream:
    def __init__(self, source, name):
        self.source = source
        self.name = name
        self.cap = None
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self.status = "connecting"
        self.error = ""
        self.thread = None

    def start(self):
        if self.running:
            return

        self.running = True
        self.status = "connecting"
        self.thread = threading.Thread(
            target=self._capture_loop,
            daemon=True
        )
        self.thread.start()

    def _capture_loop(self):
        while self.running:
            try:
                if isinstance(self.source, int):
                    cap = cv2.VideoCapture(
                        self.source, cv2.CAP_DSHOW
                    )
                else:
                    cap = cv2.VideoCapture(
                        self.source, cv2.CAP_FFMPEG
                    )

                self.cap = cap

                if not cap.isOpened():
                    self.status = "reconnecting"
                    self.error = "Could not open camera"
                    cap.release()
                    time.sleep(2)
                    continue

                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                failures = 0

                while self.running:
                    success, frame = cap.read()

                    if success and frame is not None:
                        frame = cv2.cvtColor(
                            frame, cv2.COLOR_BGR2RGB
                        )

                        with self.lock:
                            self.frame = frame

                        self.status = "live"
                        self.error = ""
                        failures = 0
                    else:
                        failures += 1
                        if failures >= 20:
                            break
                        time.sleep(0.05)

            except Exception as error:
                self.error = str(error)

            finally:
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None

            if self.running:
                self.status = "reconnecting"
                time.sleep(2)

        self.status = "stopped"

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def stop(self):
        self.running = False

        if self.cap is not None:
            self.cap.release()

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3)

        with self.lock:
            self.frame = None

        self.status = "stopped"


class MultiCameraManager:
    def __init__(self):
        self.cameras = {}

    def connect(self, number, source):
        self.disconnect(number)

        camera = CameraStream(
            source,
            f"Camera {number}"
        )

        self.cameras[number] = camera
        camera.start()

    def disconnect(self, number):
        camera = self.cameras.pop(number, None)

        if camera:
            camera.stop()

    def stop_all(self):
        for number in list(self.cameras.keys()):
            self.disconnect(number)


def parse_camera_source(value):
    value = value.strip()

    if value.isdigit():
        return int(value)

    return value


@st.fragment(run_every=0.5)
def render_extra_cameras(manager):
    st.subheader("Additional Camera Feeds")

    if not manager.cameras:
        st.info("No additional cameras connected.")
        return

    numbers = sorted(manager.cameras.keys())
    columns = st.columns(2)

    for index, number in enumerate(numbers):
        camera = manager.cameras[number]

        with columns[index % 2]:
            st.markdown(f"**Camera {number}**")

            frame = camera.get_frame()

            if frame is not None:
                st.image(frame, width="stretch")
                st.caption("LIVE")
            else:
                st.warning(
                    f"Status: {camera.status}"
                )

                if camera.error:
                    st.caption(camera.error)
