"""Independent camera capture streams for Digital Eye camera slots 2-4.

Camera 1 continues to use the existing CameraReader recording pipeline.
This manager currently provides independent live capture and reconnect for
additional cameras; shared AI and recording integration is a later stage.
"""
from __future__ import annotations

import threading
import time
from typing import Any

import cv2
import streamlit as st

from camera_config import CameraConfigStore, parse_camera_source
from settings import RECONNECT_WAIT


class CameraStream:
    def __init__(self, source: Any, name: str, width: int = 640,
                 height: int = 480, fps: int = 15):
        self.source = source
        self.name = name
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.frame = None
        self.lock = threading.Lock()
        self.cap_lock = threading.Lock()
        self.running = False
        self.status = "disconnected"
        self.error = ""
        self.thread = None

    def start(self):
        if self.running and self.thread and self.thread.is_alive():
            return
        self.running = True
        self.status = "connecting"
        self.error = ""
        self.thread = threading.Thread(target=self._capture_loop, daemon=True,
                                       name=f"DigitalEye-{self.name}")
        self.thread.start()

    def _open_capture(self):
        if isinstance(self.source, int):
            capture = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        else:
            capture = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_FPS, self.fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    def _capture_loop(self):
        while self.running:
            capture = None
            try:
                self.status = "connecting" if self.frame is None else "reconnecting"
                capture = self._open_capture()
                with self.cap_lock:
                    self.cap = capture
                if not capture.isOpened():
                    raise RuntimeError("Could not open camera source")

                failures = 0
                while self.running:
                    ok, frame = capture.read()
                    if ok and frame is not None:
                        with self.lock:
                            self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        self.status = "live"
                        self.error = ""
                        failures = 0
                    else:
                        failures += 1
                        if failures >= 20:
                            raise RuntimeError("Camera stopped sending frames")
                        time.sleep(0.05)
            except Exception as exc:
                self.error = str(exc)
                self.status = "reconnecting" if self.running else "stopped"
            finally:
                with self.cap_lock:
                    if self.cap is capture:
                        self.cap = None
                if capture is not None:
                    capture.release()
            if self.running:
                time.sleep(RECONNECT_WAIT)
        self.status = "stopped"

    def get_frame(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        with self.cap_lock:
            capture = self.cap
            self.cap = None
        if capture is not None:
            capture.release()
        if self.thread and self.thread.is_alive() and self.thread is not threading.current_thread():
            self.thread.join(timeout=4)
        with self.lock:
            self.frame = None
        self.status = "stopped"


class MultiCameraManager:
    def __init__(self, config_store: CameraConfigStore | None = None):
        self.cameras: dict[int, CameraStream] = {}
        self.config_store = config_store or CameraConfigStore()
        self._lock = threading.RLock()

    def connect(self, number: int, source: Any | None = None, name: str | None = None):
        if isinstance(number, bool) or not isinstance(number, int) or number not in (2, 3, 4):
            raise ValueError("Additional camera number must be 2, 3 or 4")
        config = self.config_store.get(number)
        selected_source = config["source"] if source is None else source
        selected_name = config["name"] if name is None else str(name).strip()
        if selected_source == "" or selected_source is None:
            raise ValueError("Camera source cannot be empty")
        if not selected_name:
            raise ValueError("Camera name cannot be empty")
        self.disconnect(number)
        self.config_store.save(number, source=str(selected_source), name=selected_name, enabled=True)
        camera = CameraStream(selected_source, selected_name, config["width"], config["height"], config["fps"])
        with self._lock:
            self.cameras[number] = camera
        camera.start()
        return camera

    def disconnect(self, number: int, clear_saved: bool = False):
        with self._lock:
            camera = self.cameras.pop(number, None)
        if camera:
            camera.stop()
        if clear_saved:
            self.config_store.save(number, enabled=False)

    def restore_enabled(self):
        """Reconnect previously enabled additional camera slots."""
        restored = []
        for number in (2, 3, 4):
            config = self.config_store.get(number)
            with self._lock:
                active = self.cameras.get(number)
            if active and active.running:
                continue
            if config["enabled"] and config["source"]:
                try:
                    self.connect(number)
                    restored.append(number)
                except (ValueError, OSError):
                    self.config_store.save(number, enabled=False)
        return restored

    def stop_all(self, preserve_configuration: bool = True):
        with self._lock:
            numbers = list(self.cameras)
        for number in numbers:
            self.disconnect(number, clear_saved=not preserve_configuration)


@st.fragment(run_every=0.5)
def render_extra_cameras(manager):
    active = manager.cameras
    if not active:
        st.info("No additional cameras are currently connected.")
        return
    columns = st.columns(2)
    for index, number in enumerate(sorted(active)):
        camera = active[number]
        with columns[index % 2]:
            st.markdown(f"**{camera.name}**")
            frame = camera.get_frame()
            if frame is not None:
                st.image(frame, width="stretch")
                st.caption(f"LIVE · Camera {number}")
            else:
                st.warning(f"{camera.status.title()}: {camera.error or 'Waiting for frames'}")
