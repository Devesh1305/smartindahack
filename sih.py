import cv2
import time
from ultralytics import YOLO

# Load YOLOv11 model (nano = fastest)
model = YOLO("yolo11n.pt")

# Open USB camera (0 = default, change if needed)
cap = cv2.VideoCapture(0)

# Reduce resolution for smoother feed
cap.set(3, 640)  # width
cap.set(4, 480)  # height

prev_area = None
last_box = None
last_status = "Detecting..."

frame_count = 0
skip_rate = 2   # start with YOLO every 2nd frame

# FPS measurement
last_time = time.time()
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Measure FPS
    current_time = time.time()
    elapsed = current_time - last_time
    if elapsed > 1:
        fps = frame_count / elapsed
        frame_count = 0
        last_time = current_time

        # Adjust skip rate dynamically based on FPS
        if fps < 8:          # too slow → skip more frames
            skip_rate = min(skip_rate + 1, 5)
        elif fps > 15:       # smooth enough → skip fewer frames
            skip_rate = max(skip_rate - 1, 1)

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

    # Draw last known box + status
    if last_box is not None:
        x1, y1, x2, y2 = last_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, last_status, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show FPS on screen
    cv2.putText(frame, f"FPS: {fps:.1f} | Skip: {skip_rate}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("YOLO Object Distance Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
