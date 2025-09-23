import cv2
from ultralytics import YOLO
import threading
from dronekit import connect, VehicleMode
import time

# --- Connect to Pixhawk 6X via TELEM2 UART ---
# Use '/dev/serial0' for GPIO14/15 UART connection
vehicle = connect('/dev/serial0', baud=57600, wait_ready=True)

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

prev_area = None
last_box = None
last_status = "Detecting..."
lock = threading.Lock()

# Threshold area (from your trigger frame)
THRESHOLD_AREA = 40000

def detect(frame_small):
    global last_box, last_status, prev_area
    results = model(frame_small, imgsz=320, verbose=False)
    for r in results:
        boxes = r.boxes
        if len(boxes) > 0:
            biggest_box = max(
                boxes,
                key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1])
            )
            x1, y1, x2, y2 = biggest_box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            area = (x2 - x1) * (y2 - y1)

            if area > THRESHOLD_AREA:
                status = "Too Close → Going Back"
                go_back()
            else:
                status = "Safe distance"

            prev_area = area
            with lock:
                last_box = (x1, y1, x2, y2)
                last_status = status

def go_back():
    """Send velocity command to Pixhawk 6X to move backward"""
    if vehicle.mode.name != "GUIDED":
        vehicle.mode = VehicleMode("GUIDED")
        time.sleep(1)

    # -0.3 m/s backward (tune speed as needed)
    send_body_velocity(-0.3, 0, 0)

def send_body_velocity(vx, vy, vz):
    """Send velocity command in local NED frame"""
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,  # time_boot_ms
        0, 0,  # target system, target component
        1,  # MAV_FRAME_LOCAL_NED
        0b0000111111000111,  # type_mask (only velocities enabled)
        0, 0, 0,  # x, y, z positions
        vx, vy, vz,  # velocities in m/s
        0, 0, 0,  # accelerations
        0, 0)  # yaw, yaw_rate
    vehicle.send_mavlink(msg)
    vehicle.flush()

# --- Main loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_small = cv2.resize(frame, (320, 240))
    threading.Thread(target=detect, args=(frame_small.copy(),), daemon=True).start()

    with lock:
        if last_box is not None:
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
vehicle.close()
