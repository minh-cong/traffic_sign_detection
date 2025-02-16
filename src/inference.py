import cv2
import time
from ultralytics import YOLO

model = YOLO("best.onnx")

cap = cv2.VideoCapture(0)
prev_time = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    results = model.predict(frame, imgsz=640, conf=0.5, device="cuda")

    annotated_frame = results[0].plot()
    cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Traffic Sign Detection", annotated_frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
