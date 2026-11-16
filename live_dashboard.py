import os
import re
import time
import threading

import cv2
import streamlit as st

from datetime import datetime

from settings import (
    CAMERA_NAME,
    INCIDENTS_DIR,
    GRACE,
    USE_FEET,
    PERSON_EVERY,
    RESTRICTED_COLOR,
    WARNING_COLOR,
    FONT,
    WARNING_SENDS_EMAIL,
    CLIP_BEFORE,
    CLIP_AFTER
)

from helpers import fmt_time, put_label
from database import add_incident, alerts_today, recent_incidents
from zones import load_zones, zone_color, draw_zone_box
from email_alerts import send_email
from detection import detect_people


# Show previously saved alerts in the dashboard.
def render_history(history_placeholder):
    rows = recent_incidents()

    with history_placeholder.container():
        if not rows:
            st.info("No alerts yet.")
            return

        columns = st.columns(3)

        for index, (
            timestamp,
            seconds,
            image_path,
            alert_type,
            details,
            clip_path
        ) in enumerate(rows):

            if alert_type == "warning":
                caption = (
                    f"{timestamp} | WARNING - "
                    f"{details} | {seconds}s"
                )

            elif alert_type == "zone":
                caption = (
                    f"{timestamp} | "
                    f"{details or 'Zone'} | {seconds}s in zone"
                )

            else:
                caption = (
                    f"{timestamp} | Previous incident: "
                    f"{details or 'Saved incident'}"
                )

            with columns[index % 3]:
                if image_path and os.path.exists(image_path):
                    st.image(
                        image_path,
                        caption=caption,
                        width=300
                    )
                else:
                    st.write(caption + " (photo missing)")

                if clip_path:
                    if os.path.exists(clip_path):
                        with st.expander("Play video clip"):
                            st.video(clip_path)
                    else:
                        st.caption("Video clip is being saved...")


# Calculate how long a person has occupied a particular zone.
def get_zone_dwell(zone_state, zone_id, now_seconds):
    start = zone_state[zone_id]["start"]

    if start is None:
        return 0

    return now_seconds - start


# --------------------------------------------------
# LIVE CAMERA FRAGMENT
# --------------------------------------------------
@st.fragment(run_every=0.1)
def live_fragment(
    reader,
    model,
    repeat_seconds,
    detection_size,
    show_outlines
):
    state = st.session_state

    # Store values between fragment refreshes.
    state.setdefault("live_last_id", -1)
    state.setdefault("live_frame_no", 0)
    state.setdefault("live_person_boxes", [])
    state.setdefault("live_fps_count", 0)
    state.setdefault("live_fps_start", time.time())
    state.setdefault("live_last_metrics", None)
    state.setdefault("live_camera_fps", 0.0)
    state.setdefault("live_status", "Starting")
    state.setdefault("live_zones_text", "-")
    state.setdefault("live_alert_count", alerts_today())
    state.setdefault("live_last_history", 0.0)

    # Update dashboard metrics.
    st.session_state["people_metric"].metric(
        "People now", len(state["live_person_boxes"])
    )

    st.session_state["status_metric"].metric(
        "Status", state["live_status"]
    )

    st.session_state["zones_metric"].metric(
        "Zones occupied", state["live_zones_text"]
    )

    st.session_state["alerts_metric"].metric(
        "Alerts today", state["live_alert_count"]
    )

    st.session_state["camera_metric"].metric(
        "Camera",
        "LIVE" if reader.status == "live" else reader.status.title(),
        f"{state['live_camera_fps']:.0f} FPS",
        delta_color="off"
    )

    # Load enabled zones only.
    zones = [
        zone for zone in load_zones()
        if zone.get("enabled", True)
    ]

    zone_ids = tuple(zone["id"] for zone in zones)

    # Reset timers if the enabled zones change.
    if state.get("live_zone_ids") != zone_ids:
        state["live_zone_ids"] = zone_ids

        state["live_zone_state"] = {
            zone_id: {
                "start": None,
                "last_inside": 0.0
            }
            for zone_id in zone_ids
        }

    zone_state = state["live_zone_state"]

    if not zones:
        st.info(
            "No zones yet. Choose 'Edit zones' in the sidebar "
            "to draw a restricted or warning area."
        )

    # Get the newest camera frame.
    frame_id, frame = reader.get_frame(state["live_last_id"])
    now_seconds = time.time()

    # Refresh incident history every two seconds.
    if now_seconds - state["live_last_history"] >= 2.0:
        render_history(st.session_state["history_placeholder"])
        state["live_last_history"] = now_seconds

    # Handle camera connection problems.
    if frame is None:
        if reader.status in ("connecting", "reconnecting"):
            st.session_state["video_placeholder"].info(
                "Starting the camera..."
                if reader.status == "connecting"
                else "Camera disconnected. Reconnecting automatically..."
            )
        else:
            st.session_state["video_placeholder"].warning(
                "No camera frame is available."
            )

        if reader.recording:
            st.session_state["recording_placeholder"].error(
                f"● Recording {fmt_time(reader.rec_elapsed())}"
            )
        else:
            st.session_state["recording_placeholder"].empty()

        return

    state["live_last_id"] = frame_id
    state["live_frame_no"] += 1
    state["live_fps_count"] += 1

    frame_no = state["live_frame_no"]
    height, width = frame.shape[:2]

    # Convert zone percentage coordinates into camera pixel coordinates.
    zone_pixels = [
        (
            zone,
            int(width * zone["x1"]),
            int(height * zone["y1"]),
            int(width * zone["x2"]),
            int(height * zone["y2"])
        )
        for zone in zones
    ]

    # Track whether each zone currently contains a person.
    occupied = {
        zone["id"]: False
        for zone in zones
    }

    # Run person detection on the first frame and then every
    # PERSON_EVERY frames.
    if frame_no == 1 or frame_no % PERSON_EVERY == 0:
        state["live_person_boxes"] = detect_people(
            model,
            frame,
            detection_size
        )

    person_boxes = state["live_person_boxes"]
    detections = []

    # Check each person's centre or feet against all enabled zones.
    for x1, y1, x2, y2 in person_boxes:
        px = (x1 + x2) // 2
        py = y2 if USE_FEET else (y1 + y2) // 2

        in_red = False
        in_warning = False

        for zone, zx1, zy1, zx2, zy2 in zone_pixels:
            if zx1 <= px <= zx2 and zy1 <= py <= zy2:
                occupied[zone["id"]] = True

                if zone["type"] == "Restricted":
                    in_red = True
                else:
                    in_warning = True

        detections.append((
            x1, y1, x2, y2,
            px, py,
            in_red,
            in_warning
        ))

    people = len(person_boxes)

    # Check whether any Restricted or Warning Zone is occupied.
    any_red = any(
        occupied[zone["id"]]
        for zone in zones
        if zone["type"] == "Restricted"
    )

    any_warning = any(
        occupied[zone["id"]]
        for zone in zones
        if zone["type"] != "Restricted"
    )

    # Update the dwell timer for every zone.
    for zone, *_ in zone_pixels:
        zone_id = zone["id"]
        zone_info = zone_state[zone_id]

        if occupied[zone_id]:
            zone_info["last_inside"] = now_seconds

            if zone_info["start"] is None:
                zone_info["start"] = now_seconds

        elif (
            zone_info["start"] is not None
            and now_seconds - zone_info["last_inside"] > GRACE
        ):
            zone_info["start"] = None
            state[f"last_alert_zone_{zone_id}"] = 0.0

    # Tint zones that currently contain a person.
    overlay = frame.copy()
    tinted = False

    for zone, x1, y1, x2, y2 in zone_pixels:
        if occupied[zone["id"]]:
            cv2.rectangle(
                overlay,
                (x1, y1),
                (x2, y2),
                zone_color(zone),
                -1
            )
            tinted = True

    if tinted:
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

    # Draw zone borders and labels.
    for zone in zones:
        if occupied[zone["id"]]:
            draw_zone_box(
                frame, zone, width, height, active=True
            )

        elif show_outlines:
            draw_zone_box(
                frame, zone, width, height, active=False
            )

    # Draw person bounding boxes.
    for (
        x1, y1, x2, y2,
        px, py,
        in_red,
        in_warning
    ) in detections:

        if in_red:
            color = RESTRICTED_COLOR
            label = "INTRUDER"

        elif in_warning:
            color = WARNING_COLOR
            label = "WARNING"

        else:
            color = (0, 255, 0)
            label = "Person"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, (px, py), 6, color, -1)

        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            FONT,
            0.7,
            color,
            2
        )

    now = datetime.now()

    # Add timestamp to the live image.
    cv2.putText(
        frame,
        now.strftime("%Y-%m-%d %H:%M:%S"),
        (10, height - 15),
        FONT,
        0.6,
        (255, 255, 255),
        2
    )

    # Display dwell countdowns for occupied zones.
    text_y = 30

    for zone, *_ in zone_pixels:
        if occupied[zone["id"]]:
            dwell = get_zone_dwell(
                zone_state,
                zone["id"],
                now_seconds
            )

            put_label(
                frame,
                f"{zone['name']}: {int(dwell)}s / {zone['timer']}s",
                (10, text_y),
                zone_color(zone)
            )

            text_y += 28

    # Show recording indicator on the camera image.
    if reader.recording:
        cv2.circle(
            frame,
            (width - 100, 22),
            7,
            (0, 0, 255),
            -1
        )

        put_label(
            frame,
            f"REC {fmt_time(reader.rec_elapsed())}",
            (width - 88, 29),
            scale=0.55
        )

    # Update dashboard status.
    if any_red:
        status = "ALERT"

    elif any_warning:
        status = "WARNING"

    else:
        status = "SAFE"

    occupied_count = sum(occupied.values())
    zones_text = f"{occupied_count} / {len(zones)}"

    metrics = (people, status, zones_text)

    state["live_status"] = status
    state["live_zones_text"] = zones_text

    if metrics != state["live_last_metrics"]:
        st.session_state["people_metric"].metric(
            "People now", people
        )

        st.session_state["status_metric"].metric(
            "Status", status
        )

        st.session_state["zones_metric"].metric(
            "Zones occupied", zones_text
        )

        state["live_last_metrics"] = metrics

    # Display the newest annotated frame.
    st.session_state["video_placeholder"].image(
        frame,
        channels="BGR",
        output_format="JPEG",
        width=720
    )

    # Calculate dashboard processing FPS.
    # This is not the camera's hardware FPS.
    if now_seconds - state["live_fps_start"] >= 1.0:
        elapsed = max(
            now_seconds - state["live_fps_start"],
            0.001
        )

        state["live_camera_fps"] = (
            state["live_fps_count"] / elapsed
        )

        state["live_fps_count"] = 0
        state["live_fps_start"] = now_seconds

    st.session_state["camera_metric"].metric(
        "Camera",
        "LIVE",
        f"{state['live_camera_fps']:.0f} FPS",
        delta_color="off"
    )

    # Refresh history when a video clip finishes saving.
    finished = reader.pop_finished_clips()

    if finished:
        render_history(st.session_state["history_placeholder"])
        state["live_last_history"] = now_seconds

    # Keep recording timer visible in the sidebar.
    if reader.recording:
        st.session_state["recording_placeholder"].error(
            f"● Recording {fmt_time(reader.rec_elapsed())}"
        )

    else:
        st.session_state["recording_placeholder"].empty()

    # Create zone incidents after the configured dwell time is reached.
    for zone, *_ in zone_pixels:
        zone_id = zone["id"]

        if not occupied[zone_id]:
            continue

        dwell = get_zone_dwell(
            zone_state,
            zone_id,
            now_seconds
        )

        alert_key = f"last_alert_zone_{zone_id}"

        if (
            dwell >= zone["timer"]
            and now_seconds - state.get(alert_key, 0.0) > repeat_seconds
        ):
            state[alert_key] = now_seconds
            restricted = zone["type"] == "Restricted"

            stamp = now.strftime("%Y%m%d_%H%M%S")

            safe_name = re.sub(
                r"[^A-Za-z0-9_-]+",
                "_",
                zone["name"]
            )[:30]

            image_path = os.path.join(
                INCIDENTS_DIR,
                f"{stamp}_{safe_name}.jpg"
            )

            snapshot = frame.copy()

            put_label(
                snapshot,
                f"ZONE: {zone['name']} ({zone['type']}) - {int(dwell)}s",
                (10, height - 45),
                zone_color(zone)
            )

            cv2.imwrite(image_path, snapshot)

            clip_path = reader.request_clip(zone["name"])

            add_incident(
                now.strftime("%Y-%m-%d %H:%M:%S"),
                people,
                int(dwell),
                image_path,
                "zone" if restricted else "warning",
                zone["name"],
                clip_path
            )

            # Restricted-zone alerts always send email.
            # Warning-zone emails depend on settings.py.
            if restricted or WARNING_SENDS_EMAIL:
                body = (
                    "DIGITAL EYE ALERT\n\n"
                    f"A person stayed {int(dwell)} seconds "
                    f"in zone: {zone['name']} ({zone['type']}).\n"
                    f"Camera: {CAMERA_NAME}\n"
                    f"Time: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    "Photo is attached.\n"
                    f"Video clip: {os.path.basename(clip_path)}\n"
                    f"Clip includes {CLIP_BEFORE}s before and "
                    f"{CLIP_AFTER}s after the alert."
                )

                threading.Thread(
                    target=send_email,
                    args=(
                        f"Digital Eye ALERT: person in {zone['name']}",
                        body,
                        image_path
                    ),
                    daemon=True
                ).start()

            # Refresh alert count and incident history.
            state["live_alert_count"] = alerts_today()

            st.session_state["alerts_metric"].metric(
                "Alerts today",
                state["live_alert_count"]
            )

            render_history(
                st.session_state["history_placeholder"]
            )


# --------------------------------------------------
# BUILD THE LIVE DASHBOARD
# --------------------------------------------------
def run_live_dashboard(
    reader,
    model,
    repeat_seconds,
    detection_size,
    show_outlines
):
    st.title("Digital Eye - Live Security Dashboard")

    col1, col2, col3, col4, col5 = st.columns(5)

    # Store placeholders so the live fragment can update them.
    st.session_state["people_metric"] = col1.empty()
    st.session_state["status_metric"] = col2.empty()
    st.session_state["zones_metric"] = col3.empty()
    st.session_state["alerts_metric"] = col4.empty()
    st.session_state["camera_metric"] = col5.empty()

    st.session_state["video_placeholder"] = st.empty()

    st.subheader("Recent alerts")
    st.session_state["history_placeholder"] = st.empty()

    st.session_state["recording_placeholder"] = st.sidebar.empty()

    # Initial metric values.
    st.session_state["people_metric"].metric("People now", "-")
    st.session_state["status_metric"].metric("Status", "Starting")
    st.session_state["zones_metric"].metric("Zones occupied", "-")

    st.session_state["alerts_metric"].metric(
        "Alerts today",
        alerts_today()
    )

    st.session_state["camera_metric"].metric(
        "Camera",
        "Starting"
    )

    render_history(
        st.session_state["history_placeholder"]
    )

    # Start camera if it is not already running.
    if not reader.running:
        reader.start()

    # Begin periodic live processing.
    live_fragment(
        reader,
        model,
        repeat_seconds,
        detection_size,
        show_outlines
    )