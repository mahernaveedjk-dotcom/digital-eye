"""Per-camera zone editor for Digital Eye.

Camera 1 uses the existing CameraReader. Cameras 2-4 are read from the
MultiCameraManager streams when available. Zone coordinates are normalized
(0..1), so they remain valid when the preview size changes.
"""

import uuid

import cv2
import streamlit as st

from zones import load_zones, save_zones, draw_zone_box

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except ImportError:
    streamlit_image_coordinates = None


_CAMERA_LABELS = {1: "Camera 1", 2: "Camera 2", 3: "Camera 3", 4: "Camera 4"}


def _get_manager_stream(manager, camera_id):
    """Find a camera stream without depending on one manager attribute name."""
    if manager is None:
        return None

    for method_name in ("get_stream", "get_camera", "get"):
        method = getattr(manager, method_name, None)
        if callable(method):
            try:
                result = method(camera_id)
                if result is not None:
                    return result
            except Exception:
                pass
            try:
                result = method(_CAMERA_LABELS[camera_id])
                if result is not None:
                    return result
            except Exception:
                pass

    for attr_name in ("streams", "cameras", "readers", "camera_streams", "_streams"):
        collection = getattr(manager, attr_name, None)
        if isinstance(collection, dict):
            for key in (camera_id, str(camera_id), _CAMERA_LABELS[camera_id], f"camera_{camera_id}"):
                if key in collection:
                    return collection[key]

    return None


def _read_frame(camera_id, reader, manager):
    """Return an RGB frame, or None when the selected camera has no frame."""
    if camera_id == 1:
        if reader is None:
            return None
        try:
            if not getattr(reader, "running", False):
                reader.start()
            result = reader.get_frame()
            if isinstance(result, tuple):
                frame = result[-1]
            else:
                frame = result
            if frame is None:
                return None
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception:
            return None

    stream = _get_manager_stream(manager, camera_id)
    if stream is None:
        return None
    try:
        result = stream.get_frame() if callable(getattr(stream, "get_frame", None)) else getattr(stream, "frame", None)
        if isinstance(result, tuple):
            result = result[-1]
        if result is None:
            return None
        frame = result.copy()
        # CameraStream.get_frame() in the current multi-camera pipeline returns RGB.
        if getattr(stream, "frame_is_bgr", False):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    except Exception:
        return None


def _zone_preview(frame, zones):
    preview = frame.copy()
    height, width = preview.shape[:2]
    for zone in zones:
        x1 = int(float(zone.get("x1", 0)) * width)
        y1 = int(float(zone.get("y1", 0)) * height)
        x2 = int(float(zone.get("x2", 0)) * width)
        y2 = int(float(zone.get("y2", 0)) * height)
        color = (255, 190, 0) if str(zone.get("type", "")).lower() == "warning" else (255, 40, 40)
        if not zone.get("enabled", True):
            color = (150, 150, 150)
        cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)
        cv2.putText(preview, str(zone.get("name", "Zone")), (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return preview


def edit_zones_page(reader, camera_manager=None):
    """Display and edit zones for Camera 1-4.

    The existing one-argument call remains compatible; pass camera_manager
    as the second argument to enable editing for cameras 2-4.
    """
    st.subheader("Zone Management")
    camera_label = st.selectbox("Select camera", list(_CAMERA_LABELS.values()), key="zone_camera_selector")
    camera_id = next(number for number, label in _CAMERA_LABELS.items() if label == camera_label)
    state_key = f"zone_editor_{camera_id}"

    zones = load_zones(camera_id)
    frame = _read_frame(camera_id, reader, camera_manager)

    if frame is None:
        if camera_id == 1:
            st.warning("Camera 1 is not providing a frame. Check the camera connection and try again.")
        else:
            st.warning(f"{camera_label} is not available. Connect and enable it in Camera Settings first.")
        st.info("You can still manage the saved zones below.")
    else:
        st.caption("Select two points on the camera image: first the top-left corner, then the bottom-right corner.")
        if streamlit_image_coordinates is None:
            st.error("Install streamlit-image-coordinates to draw zones: pip install streamlit-image-coordinates")
        else:
            points = st.session_state.setdefault(state_key, [])
            image = _zone_preview(frame, zones)
            clicked = streamlit_image_coordinates(image, key=f"zone_image_{camera_id}")
            if clicked and "x" in clicked and "y" in clicked:
                marker = (int(clicked["x"]), int(clicked["y"]))
                if not points or points[-1] != marker:
                    points.append(marker)
                    st.session_state[state_key] = points[-2:]

            points = st.session_state.get(state_key, [])
            if len(points) == 1:
                st.info("First corner selected. Click the opposite corner.")
            elif len(points) >= 2:
                (ax, ay), (bx, by) = points[-2], points[-1]
                h, w = frame.shape[:2]
                x1, x2 = sorted((max(0, min(w, ax)) / w, max(0, min(w, bx)) / w))
                y1, y2 = sorted((max(0, min(h, ay)) / h, max(0, min(h, by)) / h))
                if x2 - x1 > 0.005 and y2 - y1 > 0.005:
                    st.session_state[f"zone_draft_{camera_id}"] = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                    st.success("Area selected. Complete the zone details below and save it.")
                else:
                    st.warning("The selected area is too small. Select two different corners.")
            if st.button("Clear selected points", key=f"zone_clear_points_{camera_id}"):
                st.session_state[state_key] = []
                st.session_state.pop(f"zone_draft_{camera_id}", None)
                st.rerun()

    draft = st.session_state.get(f"zone_draft_{camera_id}")
    with st.expander("Add a new zone", expanded=bool(draft)):
        if not draft:
            st.caption("Select two corners on the camera image above to define the zone area.")
        with st.form(f"add_zone_form_{camera_id}"):
            name = st.text_input("Zone name", value="Restricted Area")
            zone_type = st.selectbox("Zone type", ["Restricted", "Warning"])
            timer = st.number_input("Alert delay (seconds)", min_value=0.0, max_value=3600.0, value=0.0, step=0.5)
            add_clicked = st.form_submit_button("Add zone", disabled=not bool(draft))
        if add_clicked and draft:
            zones.append({
                "id": uuid.uuid4().hex[:10],
                "name": name.strip() or f"Zone {len(zones) + 1}",
                "type": zone_type,
                "timer": float(timer),
                "enabled": True,
                **draft,
            })
            save_zones(zones, camera_id)
            st.session_state[state_key] = []
            st.session_state.pop(f"zone_draft_{camera_id}", None)
            st.success("Zone saved.")
            st.rerun()

    st.markdown("#### Saved zones")
    if not zones:
        st.caption("No zones have been created for this camera yet.")
        return

    changed = False
    for index, zone in enumerate(zones):
        zone_id = zone.get("id", str(index))
        with st.expander(f"{zone.get('name', f'Zone {index + 1}')} · {zone.get('type', 'Restricted')}", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                new_name = st.text_input("Name", value=str(zone.get("name", "Zone")), key=f"zone_name_{camera_id}_{zone_id}")
                new_type = st.selectbox("Type", ["Restricted", "Warning"], index=0 if str(zone.get("type", "Restricted")).lower() == "restricted" else 1, key=f"zone_type_{camera_id}_{zone_id}")
            with col2:
                new_timer = st.number_input("Alert delay (seconds)", min_value=0.0, max_value=3600.0, value=float(zone.get("timer", 0) or 0), step=0.5, key=f"zone_timer_{camera_id}_{zone_id}")
                new_enabled = st.checkbox("Enabled", value=bool(zone.get("enabled", True)), key=f"zone_enabled_{camera_id}_{zone_id}")
            if st.button("Update zone", key=f"zone_update_{camera_id}_{zone_id}"):
                zone.update({"name": new_name.strip() or zone.get("name", "Zone"), "type": new_type, "timer": float(new_timer), "enabled": bool(new_enabled)})
                changed = True
            if st.button("Delete zone", key=f"zone_delete_{camera_id}_{zone_id}", type="secondary"):
                zones.pop(index)
                save_zones(zones, camera_id)
                st.rerun()

    if changed:
        save_zones(zones, camera_id)
        st.success("Zone settings updated.")
        st.rerun()

    if frame is not None:
        st.markdown("#### Current zone preview")
        st.image(_zone_preview(frame, zones), use_container_width=True)
