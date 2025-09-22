import cv2
from ultralytics import YOLO
import threading

# Load YOLOv11 nano model
model = YOLO("yolo11n.pt")

# Open camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Display width
cap.set(4, 480)  # Display height

prev_area = None
last_box = None
last_status = "Detecting..."

lock = threading.Lock()

# YOLO detection function (runs in background)
def detect(frame_small):
    global last_box, last_status, prev_area
    results = model(frame_small, imgsz=320, verbose=False)  # Resize frame for speed

    for r in results:
        boxes = r.boxes
        if len(boxes) > 0:
            # Take biggest box
            biggest_box = max(
                boxes,
                key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1])
            )
            x1, y1, x2, y2 = biggest_box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            area = (x2 - x1) * (y2 - y1)

            # Distance comparison
            if prev_area is not None:
                if area > prev_area * 1.1:
                    status = "Getting closer"
                elif area < prev_area * 0.9:
                    status = "Moving further"
                else:
                    status = "Stable"
            else:
                status = "Detecting..."

            prev_area = area

            # Update shared variables safely
            with lock:
                last_box = (x1, y1, x2, y2)
                last_status = status

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Create a smaller copy for YOLO to process faster
    frame_small = cv2.resize(frame, (320, 240))

    # Run detection in a separate thread
    threading.Thread(target=detect, args=(frame_small.copy(),), daemon=True).start()

    # Draw last known box + status on original high-res frame
    with lock:
        if last_box is not None:
            # Scale box coordinates to match display resolution
            x1, y1, x2, y2 = last_box
            scale_x = frame.shape[1] / 320
            scale_y = frame.shape[0] / 240
            x1_disp = int(x1 * scale_x)
            y1_disp = int(y1 * scale_y)
            x2_disp = int(x2 * scale_x)
            y2_disp = int(y2 * scale_y)

            cv2.rectangle(frame, (x1_disp, y1_disp), (x2_disp, y2_disp), (0, 255, 0), 2)
            cv2.putText(frame, last_status, (x1_disp, y1_disp - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("YOLO Object Distance Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
