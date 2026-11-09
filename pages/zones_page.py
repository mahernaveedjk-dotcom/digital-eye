import time

import cv2
import streamlit as st

from streamlit_image_coordinates import streamlit_image_coordinates

from zones import load_zones, save_zones, draw_zone_box


def take_snapshot(reader):
    was_running = reader.running

    reader.start()

    frame = None
    deadline = time.time() + 10

    while time.time() < deadline:
        _, current_frame = reader.get_frame()

        if current_frame is not None:
            frame = current_frame
            break

        time.sleep(0.1)

    if not was_running:
        reader.stop()

    return frame


def edit_zones_page(reader):
    st.title("Edit zones")

    st.caption(
        "Drag with your mouse on the picture to select an area. "
        "Then give it a name, type and timer."
    )

    if st.button("Take new snapshot") or "snapshot" not in st.session_state:
        snapshot = take_snapshot(reader)

        if snapshot is None:
            st.error(
                "Cannot read from the camera. Close other apps using it "
                "and try again."
            )
            return

        st.session_state["snapshot"] = snapshot
        st.session_state["pending"] = None

    snapshot = st.session_state["snapshot"]

    zones = load_zones()

    height, width = snapshot.shape[:2]

    base = snapshot.copy()

    for zone in zones:
        draw_zone_box(base, zone, width, height)

    rgb = cv2.cvtColor(
        base,
        cv2.COLOR_BGR2RGB
    )

    value = streamlit_image_coordinates(
        rgb,
        key="zone_draw",
        click_and_drag=True,
        width=720,
        cursor="crosshair"
    )

    # Save the selected area as image fractions
    if (
        value
        and value.get("unix_time")
        != st.session_state.get("last_drag_time")
    ):
        st.session_state["last_drag_time"] = value.get(
            "unix_time"
        )

        view_width = max(value["width"], 1)
        view_height = max(value["height"], 1)

        fx1, fx2 = sorted((
            value["x1"] / view_width,
            value["x2"] / view_width
        ))

        fy1, fy2 = sorted((
            value["y1"] / view_height,
            value["y2"] / view_height
        ))

        fx1 = max(0.0, fx1)
        fx2 = min(1.0, fx2)

        fy1 = max(0.0, fy1)
        fy2 = min(1.0, fy2)

        if fx2 - fx1 > 0.03 and fy2 - fy1 > 0.03:
            st.session_state["pending"] = (
                fx1, fy1, fx2, fy2
            )
        else:
            st.session_state["pending"] = None
            st.warning("Selected area is too small.")

    pending = st.session_state.get("pending")

    if pending:
        preview = snapshot.copy()

        cv2.rectangle(
            preview,
            (
                int(width * pending[0]),
                int(height * pending[1])
            ),
            (
                int(width * pending[2]),
                int(height * pending[3])
            ),
            (0, 255, 255),
            3
        )

        st.image(
            preview,
            channels="BGR",
            width=360,
            caption="Selected area"
        )
    else:
        st.info("Drag on the camera picture to select an area.")

    with st.form("add_zone", clear_on_submit=True):
        name = st.text_input(
            "Zone name",
            value=f"Zone {len(zones) + 1}"
        )

        zone_type = st.selectbox(
            "Type",
            ["Restricted", "Warning"]
        )

        timer = st.number_input(
            "Seconds before alert",
            1,
            120,
            10
        )

        add = st.form_submit_button("Add zone")

    if add:
        if not pending:
            st.error("Select an area first.")

        elif not name.strip():
            st.error("Please enter a zone name.")

        else:
            new_id = max(
                [zone["id"] for zone in zones],
                default=0
            ) + 1

            zones.append({
                "id": new_id,
                "name": name.strip(),
                "type": zone_type,
                "timer": int(timer),
                "enabled": True,
                "x1": pending[0],
                "y1": pending[1],
                "x2": pending[2],
                "y2": pending[3]
            })

            save_zones(zones)

            st.session_state["pending"] = None
            st.rerun()

    st.subheader("Saved zones")

    if not zones:
        st.info("No zones created yet.")

    for zone in zones:
        zone_id = zone["id"]

        with st.expander(
            f"{zone['name']} | {zone['type']} | {zone['timer']}s"
        ):
            name = st.text_input(
                "Name",
                zone["name"],
                key=f"name_{zone_id}"
            )

            zone_type = st.selectbox(
                "Type",
                ["Restricted", "Warning"],
                index=(
                    0 if zone["type"] == "Restricted" else 1
                ),
                key=f"type_{zone_id}"
            )

            timer = st.number_input(
                "Seconds before alert",
                1,
                120,
                int(zone["timer"]),
                key=f"timer_{zone_id}"
            )

            enabled = st.checkbox(
                "Active",
                value=zone.get("enabled", True),
                key=f"active_{zone_id}"
            )

            col1, col2 = st.columns(2)

            if col1.button(
                "Save changes",
                key=f"save_{zone_id}"
            ):
                zone.update({
                    "name": name.strip() or zone["name"],
                    "type": zone_type,
                    "timer": int(timer),
                    "enabled": enabled
                })

                save_zones(zones)
                st.rerun()

            if col2.button(
                "Delete zone",
                key=f"delete_{zone_id}"
            ):
                save_zones([
                    item for item in zones
                    if item["id"] != zone_id
                ])

                st.rerun()