import cv2
from ultralytics import YOLO

# Load YOLOv11 model (nano version)
model = YOLO("yolo11n.pt")

# Open camera (0 = default)
cap = cv2.VideoCapture(0)

# Set resolution
cap.set(3, 320)  # width
cap.set(4, 240)  # height

prev_area = None
frame_count = 0
skip_rate = 3  # Run YOLO on every 3rd frame

last_box = None
last_status = "Detecting..."

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Run YOLO only on every Nth frame
    if frame_count % skip_rate == 0:
        results = model(frame, verbose=False)

        for r in results:
            boxes = r.boxes
            if len(boxes) > 0:
                # Take the biggest box
                biggest_box = max(
                    boxes,
                    key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1])
                )

                x1, y1, x2, y2 = biggest_box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Compute area
                area = (x2 - x1) * (y2 - y1)

                # Compare with previous area
                if prev_area is not None:
                    if area > prev_area * 1.1:
                        last_status = "Getting closer"
                    elif area < prev_area * 0.9:
                        last_status = "Moving further"
                    else:
                        last_status = "Stable"
                else:
                    last_status = "Detecting..."

                prev_area = area
                last_box = (x1, y1, x2, y2)

    # Draw last known box + status (even on skipped frames)
    if last_box is not None:
        x1, y1, x2, y2 = last_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, last_status, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("YOLO Object Distance Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
