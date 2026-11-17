
import cv2
import time

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera could not be opened")
    exit()

print("Camera connected. Press Q to exit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to receive frame")
        time.sleep(1)
        continue

    cv2.imshow("Laptop Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
