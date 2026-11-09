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
    DISPLAY_FPS,
    DETECTION_SIZES,
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
from detection import detect_people, detect_weapons


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

            if alert_type == "weapon":
                caption = f"{timestamp} | WEAPON: {details}"

            elif alert_type == "warning":
                caption = (
                    f"{timestamp} | WARNING - "
                    f"{details} | {seconds}s"
                )

            else:
                caption = (
                    f"{timestamp} | "
                    f"{details or 'Zone'} | {seconds}s in zone"
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


@st.fragment(run_every=0.1)
def live_fragment(
    reader,
    model,
    weapon_model,
    weapon_ids,
    weapon_on,
    weapon_conf,
    weapon_every,
    weapon_confirm,
    weapon_cooldown,
    repeat_seconds,
    detection_size,
    show_outlines
):
    """Process and display one camera frame per fragment refresh."""

    state = st.session_state

    state.setdefault("live_last_id", -1)
    state.setdefault("live_frame_no", 0)
    state.setdefault("live_person_boxes", [])
    state.setdefault("live_weapons", [])
    state.setdefault("live_weapon_hits", 0)
    state.setdefault("live_weapon_last_seen", 0.0)
    state.setdefault("live_last_display", 0.0)
    state.setdefault("live_fps_count", 0)
    state.setdefault("live_fps_start", time.time())
    state.setdefault("live_last_metrics", None)
    state.setdefault("live_last_new", time.time())
    state.setdefault("live_last_history", 0.0)

    zones = [
        zone for zone in load_zones()
        if zone.get("enabled", True)
    ]

    zone_ids = tuple(zone["id"] for zone in zones)

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

    frame_id, frame = reader.get_frame(
        state["live_last_id"]
    )

    now_seconds = time.time()

    # Camera status
    if frame is None:
        if reader.status in ("connecting", "reconnecting"):
            if reader.status == "connecting":
                st.info("Starting the camera...")
            else:
                st.warning(
                    "Camera disconnected. Reconnecting automatically..."
                )

        return

    state["live_last_id"] = frame_id
    state["live_last_new"] = now_seconds
    state["live_frame_no"] += 1
    state["live_fps_count"] += 1

    frame_no = state["live_frame_no"]

    height, width = frame.shape[:2]

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

    occupied = {
        zone["id"]: False
        for zone in zones
    }

    # Person detection
    if (
        frame_no == 1
        or frame_no % PERSON_EVERY == 0
    ):
        state["live_person_boxes"] = detect_people(
            model,
            frame,
            detection_size
        )

    person_boxes = state["live_person_boxes"]

    detections = []

    for x1, y1, x2, y2 in person_boxes:
        px = (x1 + x2) // 2

        py = (
            y2
            if USE_FEET
            else (y1 + y2) // 2
        )

        in_red = False
        in_warning = False

        for (
            zone,
            zx1,
            zy1,
            zx2,
            zy2
        ) in zone_pixels:

            if (
                zx1 <= px <= zx2
                and zy1 <= py <= zy2
            ):
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

    # Weapon detection
    if (
        weapon_on
        and weapon_model is not None
        and (frame_no + 1) % weapon_every == 0
    ):
        state["live_weapons"] = detect_weapons(
            weapon_model,
            frame,
            weapon_ids,
            weapon_conf,
            person_boxes,
            zones
        )

        if state["live_weapons"]:
            state["live_weapon_hits"] += 1
        else:
            state["live_weapon_hits"] = 0

    weapons = state["live_weapons"]
    weapon_seen = bool(weapons)

    if weapon_seen:
        state["live_weapon_last_seen"] = now_seconds

    elif (
        state["live_weapon_last_seen"]
        and now_seconds - state["live_weapon_last_seen"] > 5
    ):
        state["last_weapon_alert_RED"] = 0.0
        state["last_weapon_alert_GREEN"] = 0.0
        state["live_weapon_last_seen"] = 0.0

    weapon_red = any(
        weapon[6] == "RED"
        for weapon in weapons
    )

    weapon_green = any(
        weapon[6] == "GREEN"
        for weapon in weapons
    )

    # Zone dwell timers
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

    def dwell_of(zone_id):
        start = zone_state[zone_id]["start"]

        if start is None:
            return 0

        return now_seconds - start

    # Tint only occupied zones
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
        frame = cv2.addWeighted(
            overlay,
            0.3,
            frame,
            0.7,
            0
        )

    # Draw zones
    for zone in zones:
        if occupied[zone["id"]]:
            draw_zone_box(
                frame,
                zone,
                width,
                height,
                active=True
            )

        elif show_outlines:
            draw_zone_box(
                frame,
                zone,
                width,
                height,
                active=False
            )

    # Draw people
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

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.circle(
            frame,
            (px, py),
            6,
            color,
            -1
        )

        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            FONT,
            0.7,
            color,
            2
        )

    # Draw weapons
    for (
        x1, y1, x2, y2,
        name, confidence,
        location, held, zone_name
    ) in weapons:

        color = (
            (0, 0, 255)
            if location == "RED"
            else (0, 128, 255)
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            3
        )

        cv2.putText(
            frame,
            f"WEAPON {name} {int(confidence * 100)}% ({zone_name})",
            (x1, max(y1 - 10, 60)),
            FONT,
            0.55,
            color,
            2
        )

    # Weapon warning banner
    if weapon_seen:
        best = max(
            weapons,
            key=lambda item: item[5]
        )

        if weapon_red and weapon_green:
            banner = "CRITICAL: WEAPONS IN RED + GREEN ZONES"
            banner_color = (0, 0, 255)

        elif weapon_red:
            banner = (
                f"CRITICAL: WEAPON IN RED ZONE - "
                f"{best[4]} {int(best[5] * 100)}%"
            )
            banner_color = (0, 0, 255)

        else:
            banner = (
                f"WARNING: WEAPON IN GREEN ZONE - "
                f"{best[4]} {int(best[5] * 100)}%"
            )
            banner_color = (0, 100, 230)

        cv2.rectangle(
            frame,
            (0, 0),
            (width, 45),
            banner_color,
            -1
        )

        cv2.putText(
            frame,
            banner,
            (10, 32),
            FONT,
            0.8,
            (255, 255, 255),
            2
        )

    now = datetime.now()

    cv2.putText(
        frame,
        now.strftime("%Y-%m-%d %H:%M:%S"),
        (10, height - 15),
        FONT,
        0.6,
        (255, 255, 255),
        2
    )

    # Zone countdowns
    text_y = 75 if weapon_seen else 30

    for zone, *_ in zone_pixels:
        if occupied[zone["id"]]:
            put_label(
                frame,
                f"{zone['name']}: "
                f"{int(dwell_of(zone['id']))}s / "
                f"{zone['timer']}s",
                (10, text_y),
                zone_color(zone)
            )

            text_y += 28

    # Manual recording indicator
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

    # Dashboard metrics
    if weapon_seen:
        status = "WEAPON!"
    elif any_red:
        status = "ALERT"
    elif any_warning:
        status = "WARNING"
    else:
        status = "SAFE"

    occupied_count = sum(occupied.values())

    metrics = (
        people,
        status,
        f"{occupied_count} / {len(zones)}"
    )

    if metrics != state["live_last_metrics"]:
        st.session_state["people_metric"].metric(
            "People now",
            people
        )

        st.session_state["status_metric"].metric(
            "Status",
            status
        )

        st.session_state["zones_metric"].metric(
            "Zones occupied",
            metrics[2]
        )

        state["live_last_metrics"] = metrics

    # Display video
    if (
        now_seconds - state["live_last_display"]
        >= 1.0 / DISPLAY_FPS
    ):
        state["live_last_display"] = now_seconds

        st.session_state["video_placeholder"].image(
            frame,
            channels="BGR",
            output_format="JPEG",
            width=720
        )

    # Camera speed
    if now_seconds - state["live_fps_start"] >= 1.0:
        fps = (
            state["live_fps_count"]
            / (now_seconds - state["live_fps_start"])
        )

        state["live_fps_count"] = 0
        state["live_fps_start"] = now_seconds

        st.session_state["camera_metric"].metric(
            "Camera",
            "LIVE",
            f"{fps:.0f} FPS",
            delta_color="off"
        )

        finished = reader.pop_finished_clips()

        if finished:
            render_history(
                st.session_state["history_placeholder"]
            )

        if reader.recording:
            st.session_state["recording_placeholder"].error(
                f"● Recording {fmt_time(reader.rec_elapsed())}"
            )
        else:
            st.session_state["recording_placeholder"].empty()

    # Zone alerts
    for zone, *_ in zone_pixels:
        zone_id = zone["id"]

        if not occupied[zone_id]:
            continue

        dwell = dwell_of(zone_id)

        alert_key = f"last_alert_zone_{zone_id}"

        if (
            dwell >= zone["timer"]
            and now_seconds - state.get(alert_key, 0.0)
            > repeat_seconds
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
                f"ZONE: {zone['name']} "
                f"({zone['type']}) - {int(dwell)}s",
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

            st.session_state["alerts_metric"].metric(
                "Alerts today",
                alerts_today()
            )

            render_history(
                st.session_state["history_placeholder"]
            )

    # Weapon alerts
    if (
        weapon_seen
        and state["live_weapon_hits"] >= weapon_confirm
    ):
        red_list = [
            item for item in weapons
            if item[6] == "RED"
        ]

        green_list = [
            item for item in weapons
            if item[6] == "GREEN"
        ]

        state.setdefault("last_weapon_alert_RED", 0.0)
        state.setdefault("last_weapon_alert_GREEN", 0.0)

        due_red = (
            bool(red_list)
            and now_seconds - state["last_weapon_alert_RED"]
            > weapon_cooldown
        )

        due_green = (
            bool(green_list)
            and now_seconds - state["last_weapon_alert_GREEN"]
            > weapon_cooldown
        )

        if due_red or due_green:
            if red_list:
                state["last_weapon_alert_RED"] = now_seconds

            if green_list:
                state["last_weapon_alert_GREEN"] = now_seconds

            level = "CRITICAL" if red_list else "HIGH"

            if red_list and green_list:
                location_text = "RED and GREEN zones"
            elif red_list:
                location_text = "RED (restricted) zone"
            else:
                location_text = "GREEN (safe) area"

            found = red_list + green_list

            zone_names = ", ".join(
                dict.fromkeys(item[8] for item in found)
            )

            lines = [
                (
                    f"- {item[4]} "
                    f"({int(item[5] * 100)}% confidence) "
                    f"in {item[6]} zone - {item[8]}, "
                    f"held by person: "
                    f"{'YES' if item[7] else 'not clear'}"
                )
                for item in found
            ]

            detail = (
                f"{level} | "
                + "; ".join(
                    f"{item[4]} {int(item[5] * 100)}% "
                    f"{item[6]} - {item[8]}"
                    for item in found
                )
            )

            stamp = now.strftime("%Y%m%d_%H%M%S")

            image_path = os.path.join(
                INCIDENTS_DIR,
                f"weapon_{stamp}.jpg"
            )

            snapshot = frame.copy()

            put_label(
                snapshot,
                f"WEAPON in: {zone_names}",
                (10, height - 45),
                (0, 0, 255)
            )

            cv2.imwrite(image_path, snapshot)

            clip_path = reader.request_clip("weapon")

            add_incident(
                now.strftime("%Y-%m-%d %H:%M:%S"),
                people,
                int(max(
                    [
                        dwell_of(zone["id"])
                        for zone in zones
                    ],
                    default=0
                )),
                image_path,
                "weapon",
                detail,
                clip_path
            )

            body = (
                f"DIGITAL EYE - WEAPON ALERT ({level})\n\n"
                f"Possible weapon detected in {location_text}.\n"
                + "\n".join(lines)
                + "\n\n"
                f"Person inside restricted zone: "
                f"{'YES' if any_red else 'No'}\n"
                f"People in view: {people}\n"
                f"Camera: {CAMERA_NAME}\n"
                f"Time: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
                "Photo is attached.\n"
                f"Video clip: {os.path.basename(clip_path)}\n\n"
                "This is an automatic AI detection and it can be wrong. "
                "Please check the photo before taking action."
            )

            threading.Thread(
                target=send_email,
                args=(
                    f"{level}: Weapon detected - "
                    f"{zone_names} - Digital Eye",
                    body,
                    image_path
                ),
                daemon=True
            ).start()

            st.session_state["alerts_metric"].metric(
                "Alerts today",
                alerts_today()
            )

            st.session_state["weapon_metric"].metric(
                "Weapon alerts today",
                alerts_today("weapon")
            )

            render_history(
                st.session_state["history_placeholder"]
            )


def run_live_dashboard(
    reader,
    model,
    weapon_model,
    weapon_ids,
    weapon_on,
    weapon_conf,
    weapon_every,
    weapon_confirm,
    weapon_cooldown,
    repeat_seconds,
    detection_size,
    show_outlines
):
    st.title("Digital Eye - Live Security Dashboard")

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    st.session_state["people_metric"] = col1.empty()
    st.session_state["status_metric"] = col2.empty()
    st.session_state["zones_metric"] = col3.empty()
    st.session_state["alerts_metric"] = col4.empty()
    st.session_state["weapon_metric"] = col5.empty()
    st.session_state["camera_metric"] = col6.empty()

    st.session_state["video_placeholder"] = st.empty()

    st.subheader("Recent alerts")
    st.session_state["history_placeholder"] = st.empty()

    st.session_state["recording_placeholder"] = st.sidebar.empty()

    st.session_state["people_metric"].metric(
        "People now",
        "-"
    )

    st.session_state["status_metric"].metric(
        "Status",
        "Starting"
    )

    st.session_state["zones_metric"].metric(
        "Zones occupied",
        "-"
    )

    st.session_state["alerts_metric"].metric(
        "Alerts today",
        alerts_today()
    )

    st.session_state["weapon_metric"].metric(
        "Weapon alerts today",
        alerts_today("weapon")
    )

    render_history(
        st.session_state["history_placeholder"]
    )

    if not reader.running:
        reader.start()

    live_fragment(
        reader,
        model,
        weapon_model,
        weapon_ids,
        weapon_on,
        weapon_conf,
        weapon_every,
        weapon_confirm,
        weapon_cooldown,
        repeat_seconds,
        detection_size,
        show_outlines
    )