import collections
import os
import queue
import re
import shutil
import threading
import time

import cv2
import numpy as np

from datetime import datetime

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None

from settings import (
    CAMERA_INDEX,
    RECORDINGS_DIR,
    CONTINUOUS_DIR,
    CLIPS_DIR,
    RECONNECT_WAIT,
    RECORD_FPS,
    MAX_RECORD_SECONDS,
    CHUNK_SECONDS,
    CLIP_BEFORE,
    CLIP_AFTER,
    MAX_CLIP_SECONDS,
    MIN_FREE_GB,
    FONT
)


def new_video_path(folder, prefix, suffix=""):
    name = datetime.now().strftime(
        f"{prefix}_%Y%m%d_%H%M%S"
    ) + suffix

    path = os.path.join(folder, name + ".mp4")

    number = 2
    while os.path.exists(path):
        path = os.path.join(
            folder,
            f"{name}_{number}.mp4"
        )
        number += 1

    return path


class VideoFile:
    """Writes video frames in a background thread."""

    def __init__(self, path, size, live=False, block=False):
        self.path = path
        self.size = size
        self.block = block
        self.next_write = None
        self.dropped = 0
        self.failed = False

        self.frame_queue = queue.Queue(maxsize=200)

        if imageio_ffmpeg is not None:
            params = ["-preset", "veryfast"]

            if live:
                params += [
                    "-movflags",
                    "frag_keyframe+empty_moov+default_base_moof",
                    "-tune",
                    "zerolatency",
                    "-g",
                    str(RECORD_FPS),
                    "-flush_packets",
                    "1"
                ]

            self.kind = "ffmpeg"

            self.writer = imageio_ffmpeg.write_frames(
                path,
                size,
                pix_fmt_in="bgr24",
                fps=RECORD_FPS,
                codec="libx264",
                macro_block_size=2,
                quality=6,
                output_params=params
            )

            self.writer.send(None)

        else:
            self.kind = "cv2"

            self.writer = cv2.VideoWriter(
                path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                RECORD_FPS,
                size
            )

        self.thread = threading.Thread(
            target=self._worker,
            daemon=False
        )
        self.thread.start()

    def _worker(self):
        while True:
            frame = self.frame_queue.get()

            if frame is None:
                break

            if self.failed:
                continue

            try:
                if (frame.shape[1], frame.shape[0]) != self.size:
                    frame = cv2.resize(frame, self.size)

                if self.kind == "ffmpeg":
                    self.writer.send(
                        np.ascontiguousarray(frame).tobytes()
                    )
                else:
                    self.writer.write(frame)

            except Exception as error:
                print("Video encoder problem:", error)
                self.failed = True

        try:
            if self.kind == "ffmpeg":
                self.writer.close()
            else:
                self.writer.release()

        except Exception as error:
            print("Closing video problem:", error)

    def _put(self, frame):
        if self.block:
            self.frame_queue.put(frame)
        else:
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                self.dropped += 1

    def write_timed(self, frame, when):
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
        self.frame_queue.put(None)
        self.thread.join(timeout=60)


class CameraReader:
    """Background camera reader and recording manager."""

    def __init__(self, index=CAMERA_INDEX):
        self.index = index

        self.lock = threading.Lock()
        self.wlock = threading.Lock()

        self.frame = None
        self.frame_id = 0

        self.running = False
        self.gen = 0
        self.status = "stopped"
        self.thread = None

        # Manual recording
        self.rec_flag = False
        self.rec_started = 0.0
        self.rec_path = None
        self.last_recording = None
        self.rec_error = ""
        self.manual_writer = None

        # Continuous recording
        self.continuous = False
        self.cont_blocked = False
        self.cont_error = ""
        self.cont_writer = None
        self.cont_started = 0.0
        self.disk_checked = 0.0

        # Alert clip buffer
        self.pre_buffer = collections.deque(
            maxlen=int(CLIP_BEFORE * RECORD_FPS * 2) + 10
        )

        self.buffer_next = 0.0
        self.clip = None
        self.finished_clips = []

    # -------------------------------------------------
    # Camera control
    # -------------------------------------------------

    def start(self):
        if self.running and self.thread and self.thread.is_alive():
            return

        self.running = True
        self.status = "connecting"
        self.gen += 1

        self.thread = threading.Thread(
            target=self._loop,
            args=(self.gen,),
            daemon=True
        )
        self.thread.start()

    def stop(self):
        self.running = False

        with self.wlock:
            self.rec_flag = False
            self._close_manual()
            self._close_continuous()
            self._finish_clip()

        thread = self.thread

        if thread and thread is not threading.current_thread():
            thread.join(timeout=3)

        self.thread = None
        self.status = "stopped"

        with self.lock:
            self.frame = None

    def get_frame(self, last_id=-1):
        with self.lock:
            if self.frame is None:
                return 0, None

            if self.frame_id == last_id:
                return self.frame_id, None

            return self.frame_id, self.frame.copy()

    def _loop(self, generation):
        first = True

        while self.running and self.gen == generation:
            capture = None

            try:
                self.status = (
                    "connecting" if first else "reconnecting"
                )

                capture = cv2.VideoCapture(self.index)

                if not capture.isOpened():
                    raise RuntimeError("Camera did not open")

                capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                failures = 0

                while self.running and self.gen == generation:
                    success, frame = capture.read()

                    if success and frame is not None:
                        failures = 0
                        first = False
                        self.status = "live"

                        with self.lock:
                            self.frame = frame
                            self.frame_id += 1

                        self._record(frame)

                    else:
                        failures += 1

                        with self.wlock:
                            self._check_clip(time.time())

                        if failures >= 30:
                            raise RuntimeError(
                                "Camera stopped sending frames"
                            )

                        time.sleep(0.05)

            except Exception as error:
                print("Camera problem:", error)

            finally:
                if capture is not None:
                    capture.release()

            if self.running and self.gen == generation:
                self.status = "reconnecting"
                time.sleep(RECONNECT_WAIT)

    # -------------------------------------------------
    # Recording frame processing
    # -------------------------------------------------

    def _record(self, frame):
        now = time.time()

        with self.wlock:
            stamped = frame.copy()

            timestamp = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            origin = (10, stamped.shape[0] - 12)

            cv2.putText(
                stamped, timestamp, origin,
                FONT, 0.6, (0, 0, 0), 3
            )

            cv2.putText(
                stamped, timestamp, origin,
                FONT, 0.6, (255, 255, 255), 1
            )

            try:
                self._write_manual(stamped, now)
                self._write_continuous(stamped, now)
                self._sample(stamped, now)

            except Exception as error:
                print("Recording problem:", error)

            self._check_clip(now)

    # -------------------------------------------------
    # Manual recording
    # -------------------------------------------------

    @property
    def recording(self):
        return self.rec_flag

    def rec_elapsed(self):
        if self.rec_flag:
            return time.time() - self.rec_started

        return 0.0

    def start_recording(self):
        with self.wlock:
            if self.rec_flag:
                return

            self.rec_error = ""
            self.rec_started = time.time()
            self.rec_flag = True

    def stop_recording(self):
        with self.wlock:
            self.rec_flag = False
            return self._close_manual()

    def _write_manual(self, stamped, now):
        if not self.rec_flag:
            return

        if now - self.rec_started > MAX_RECORD_SECONDS:
            self.rec_flag = False
            self._close_manual()
            self.rec_error = "Maximum recording length reached"
            return

        try:
            if self.manual_writer is None:
                height, width = stamped.shape[:2]

                self.rec_path = new_video_path(
                    RECORDINGS_DIR,
                    "rec"
                )

                self.manual_writer = VideoFile(
                    self.rec_path,
                    (width, height),
                    live=True
                )

            self.manual_writer.write_timed(stamped, now)

        except Exception as error:
            print("Manual recording problem:", error)
            self.rec_error = str(error)
            self.rec_flag = False
            self._close_manual()

    def _close_manual(self):
        path = None

        if self.manual_writer is not None:
            self.manual_writer.close()
            path = self.rec_path
            self.last_recording = path

        self.manual_writer = None
        return path

    # -------------------------------------------------
    # Continuous recording
    # -------------------------------------------------

    def _write_continuous(self, stamped, now):
        if not self.continuous or self.cont_blocked:
            if self.cont_writer is not None:
                self._close_continuous()
            return

        if now - self.disk_checked > 30:
            self.disk_checked = now

            try:
                free_space = shutil.disk_usage(
                    CONTINUOUS_DIR
                ).free
            except OSError:
                free_space = None

            if (
                free_space is not None
                and free_space < MIN_FREE_GB * 1024 ** 3
            ):
                self.cont_blocked = True

                self.cont_error = (
                    f"Continuous recording paused: less than "
                    f"{MIN_FREE_GB} GB of disk space remains."
                )

                self._close_continuous()
                return

        if (
            self.cont_writer is not None
            and now - self.cont_started >= CHUNK_SECONDS
        ):
            self._close_continuous()

        try:
            if self.cont_writer is None:
                height, width = stamped.shape[:2]

                path = new_video_path(
                    CONTINUOUS_DIR,
                    "cont"
                )

                self.cont_writer = VideoFile(
                    path,
                    (width, height),
                    live=True
                )

                self.cont_started = now

            self.cont_writer.write_timed(stamped, now)

        except Exception as error:
            self.cont_error = (
                f"Continuous recording stopped: {error}"
            )

            self.cont_blocked = True
            self._close_continuous()

    def _close_continuous(self):
        if self.cont_writer is not None:
            self.cont_writer.close()

        self.cont_writer = None

    # -------------------------------------------------
    # Alert clips
    # -------------------------------------------------

    def _sample(self, stamped, now):
        if now < self.buffer_next:
            return

        self.buffer_next = now + 0.9 / RECORD_FPS

        success, buffer = cv2.imencode(
            ".jpg",
            stamped,
            [cv2.IMWRITE_JPEG_QUALITY, 80]
        )

        if not success:
            return

        item = (now, buffer.tobytes())
        self.pre_buffer.append(item)

        if self.clip is not None:
            self.clip["frames"].append(item)

    def request_clip(self, label):
        with self.wlock:
            now = time.time()

            if self.clip is not None:
                if now - self.clip["start"] < MAX_CLIP_SECONDS:
                    self.clip["deadline"] = now + CLIP_AFTER

                return self.clip["path"]

            safe_label = re.sub(
                r"[^A-Za-z0-9_-]+",
                "_",
                str(label)
            )[:30]

            path = new_video_path(
                CLIPS_DIR,
                "clip",
                "_" + safe_label
            )

            before_frames = [
                item for item in self.pre_buffer
                if item[0] >= now - CLIP_BEFORE
            ]

            self.clip = {
                "path": path,
                "start": now,
                "deadline": now + CLIP_AFTER,
                "frames": before_frames
            }

            return path

    def _check_clip(self, now):
        if (
            self.clip is not None
            and now >= self.clip["deadline"]
        ):
            self._finish_clip()

    def _finish_clip(self):
        clip, self.clip = self.clip, None

        if clip is not None and clip["frames"]:
            threading.Thread(
                target=self._write_clip,
                args=(clip,),
                daemon=False
            ).start()

    def _write_clip(self, clip):
        try:
            frames = clip["frames"]

            first_frame = cv2.imdecode(
                np.frombuffer(frames[0][1], np.uint8),
                cv2.IMREAD_COLOR
            )

            if first_frame is None:
                return

            height, width = first_frame.shape[:2]

            video = VideoFile(
                clip["path"],
                (width, height),
                block=True
            )

            for timestamp, data in frames:
                image = cv2.imdecode(
                    np.frombuffer(data, np.uint8),
                    cv2.IMREAD_COLOR
                )

                if image is not None:
                    video.write_timed(image, timestamp)

            video.close()

            with self.wlock:
                self.finished_clips.append(clip["path"])

        except Exception as error:
            print("Saving alert clip problem:", error)

    def pop_finished_clips(self):
        with self.wlock:
            finished = self.finished_clips
            self.finished_clips = []

        return finished

    def active_paths(self):
        with self.wlock:
            paths = set()

            if self.manual_writer is not None:
                paths.add(
                    os.path.abspath(self.manual_writer.path)
                )

            if self.cont_writer is not None:
                paths.add(
                    os.path.abspath(self.cont_writer.path)
                )

            return paths