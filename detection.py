def detect_people(model, frame, image_size):
    """Detect people and return their bounding boxes."""
    try:
        results = model(
            frame,
            classes=[0],
            imgsz=image_size,
            verbose=False
        )

        boxes = [
            tuple(map(int, box.xyxy[0]))
            for box in results[0].boxes
        ]

        return boxes

    except Exception as error:
        print("Person detection problem:", error)
        return []


def detect_weapons(
    model,
    frame,
    weapon_ids,
    confidence,
    person_detections,
    zones
):
    """
    Detect guns, knives and grenades.

    RED means inside a Restricted zone.
    GREEN means outside Restricted zones.
    """

    if model is None:
        return []

    try:
        results = model(
            frame,
            classes=weapon_ids,
            conf=confidence,
            verbose=False
        )

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

        found = []

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            name = str(
                model.names[int(box.cls[0])]
            )

            confidence_value = float(box.conf[0])

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            location = "GREEN"
            zone_name = "outside all zones"

            for (
                zone,
                zx1,
                zy1,
                zx2,
                zy2
            ) in zone_pixels:

                if (
                    zx1 <= center_x <= zx2
                    and zy1 <= center_y <= zy2
                ):
                    if zone["type"] == "Restricted":
                        location = "RED"
                        zone_name = zone["name"]
                        break

                    if zone_name == "outside all zones":
                        zone_name = zone["name"]

            held = any(
                px1 <= center_x <= px2
                and py1 <= center_y <= py2
                for px1, py1, px2, py2 in person_detections
            )

            found.append((
                x1,
                y1,
                x2,
                y2,
                name,
                confidence_value,
                location,
                held,
                zone_name
            ))

        return found

    except Exception as error:
        print("Weapon detection problem:", error)
        return []