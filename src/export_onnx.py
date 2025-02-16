from ultralytics import YOLO

def export_to_onnx():
    model = YOLO("best.pt")

    model.export(
        format="onnx",
        dynamic=True,
        simplify=True,
        imgsz=640,
        opset = 12

    )

if __name__ == "__main__":
    export_to_onnx()