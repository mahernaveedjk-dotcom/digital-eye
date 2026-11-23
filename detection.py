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
