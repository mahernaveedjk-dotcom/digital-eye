import glob
import os
import shutil
import time

from datetime import datetime

import streamlit as st

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None

from settings import (
    RECORDINGS_DIR,
    CONTINUOUS_DIR,
    CLIPS_DIR,
    CLIP_BEFORE,
    CLIP_AFTER,
    CHUNK_SECONDS
)


def folder_size_mb(folder, recursive=False):
    pattern = (
        os.path.join(folder, "**", "*.mp4")
        if recursive
        else os.path.join(folder, "*.mp4")
    )

    files = glob.glob(pattern, recursive=recursive)

    return sum(
        os.path.getsize(file)
        for file in files
    ) / (1024 * 1024)


def video_browser(reader, folder, key, bulk_delete=False):
    active = reader.active_paths()

    files = sorted(
        (
            file
            for file in glob.glob(
                os.path.join(folder, "*.mp4")
            )
            if os.path.abspath(file) not in active
        ),
        key=os.path.getmtime,
        reverse=True
    )

    if not files:
        st.info("Nothing here yet.")
        return

    def video_label(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)

        created = datetime.fromtimestamp(
            os.path.getmtime(path)
        ).strftime("%Y-%m-%d %H:%M:%S")

        return (
            f"{os.path.basename(path)} | "
            f"{created} | {size_mb:.1f} MB"
        )

    choice = st.selectbox(
        "Choose a video",
        files,
        format_func=video_label,
        key=f"{key}_pick"
    )

    st.video(choice)

    col_download, col_delete = st.columns(2)

    if os.path.getsize(choice) < 100 * 1024 * 1024:
        with open(choice, "rb") as file:
            col_download.download_button(
                "Download",
                file.read(),
                file_name=os.path.basename(choice),
                mime="video/mp4",
                key=f"{key}_download"
            )
    else:
        col_download.caption(
            "Large file. Copy it directly from: " + folder
        )

    if col_delete.button(
        "Delete this video",
        key=f"{key}_delete"
    ):
        try:
            os.remove(choice)
        except OSError as error:
            st.error(f"Could not delete: {error}")

        st.rerun()

    if bulk_delete:
        st.divider()

        days = st.number_input(
            "Delete videos older than (days)",
            1,
            365,
            7,
            key=f"{key}_days"
        )

        confirm = st.checkbox(
            "I understand this cannot be undone",
            key=f"{key}_confirm"
        )

        if st.button(
            "Delete old videos",
            key=f"{key}_bulk",
            disabled=not confirm
        ):
            cutoff = time.time() - days * 86400
            removed = 0

            for file in files:
                if os.path.getmtime(file) < cutoff:
                    try:
                        os.remove(file)
                        removed += 1
                    except OSError:
                        pass

            st.success(f"Deleted {removed} video(s).")
            st.rerun()


def recordings_page(reader):
    st.title("Recordings")

    if imageio_ffmpeg is None:
        st.warning(
            "For browser-compatible MP4 recordings, install "
            "imageio-ffmpeg and record new videos."
        )

    try:
        free_gb = (
            shutil.disk_usage(RECORDINGS_DIR).free
            / 1024 ** 3
        )
    except OSError:
        free_gb = 0.0

    st.caption(
        f"Storage used: "
        f"alert clips {folder_size_mb(CLIPS_DIR):.0f} MB | "
        f"continuous {folder_size_mb(CONTINUOUS_DIR):.0f} MB | "
        f"manual {folder_size_mb(RECORDINGS_DIR):.0f} MB | "
        f"free disk {free_gb:.0f} GB"
    )

    tab_clips, tab_continuous, tab_manual = st.tabs([
        "Alert clips",
        "Continuous",
        "Manual"
    ])

    with tab_clips:
        st.caption(
            f"Each alert clip includes {CLIP_BEFORE} seconds before "
            f"and {CLIP_AFTER} seconds after the alert."
        )

        video_browser(
            reader,
            CLIPS_DIR,
            "clips"
        )

    with tab_continuous:
        st.caption(
            f"Continuous CCTV recording in "
            f"{CHUNK_SECONDS // 60}-minute files."
        )

        video_browser(
            reader,
            CONTINUOUS_DIR,
            "continuous",
            bulk_delete=True
        )

    with tab_manual:
        st.caption(
            "Recordings made using the Start/Stop recording button."
        )

        video_browser(
            reader,
            RECORDINGS_DIR,
            "manual"
        )