import cv2

from settings import FONT


def fmt_time(seconds):
    seconds = int(seconds)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def put_label(img, text, org, color=(255, 255, 255), scale=0.7):
    """Draw readable text with a dark outline."""
    cv2.putText(
        img, text, org, FONT, scale,
        (0, 0, 0), 4
    )

    cv2.putText(
        img, text, org, FONT, scale,
        color, 2
    )