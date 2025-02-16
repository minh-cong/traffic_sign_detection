from ultralytics import YOLO

model = YOLO("/home/cong/workspace/traffic_sign_detection/models/yolov8n.pt")

model.train(
    data="/home/cong/workspace/traffic_sign_detection/data/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    optimizer="AdamW",
    lr0=0.001,
    augment=True,
    hsv_h=0.2,
    degrees=15,
    flipud=0.1,
    name="traffic_sign_v1"
)
