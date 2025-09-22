import cv2
from ultralytics import YOLO

# Load YOLOv11 model (replace with your model path if needed)
model = YOLO("yolo11n.pt")  # 'n' = nano version (lightweight)

# Open camera (0 = default, change if needed)
cap = cv2.VideoCapture(0)

prev_area = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame, verbose=False)

    # Get first detection (you can modify to pick specific class)
    for r in results:
        boxes = r.boxes
        if len(boxes) > 0:
            # Take the biggest box (largest area)
            biggest_box = max(boxes, key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]))

            # Extract coordinates
            x1, y1, x2, y2 = biggest_box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Compute area of bounding box
            area = (x2 - x1) * (y2 - y1)

            # Compare with previous area
            if prev_area is not None:
                if area > prev_area * 1.1:  # 10% threshold
                    status = "Getting closer"
                elif area < prev_area * 0.9:
                    status = "Moving further"
                else:
                    status = "Stable"
            else:
                status = "Detecting..."

            prev_area = area

            # Draw box + label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, status, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("YOLO Object Distance Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
