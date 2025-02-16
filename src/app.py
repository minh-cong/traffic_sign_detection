import gradio as gr
from ultralytics import YOLO
import cv2

model = YOLO("best.onnx")

def detect(image):
    results = model.predict(image, conf=0.5)
    return results[0].plot()

iface = gr.Interface(
    fn=detect,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Image(),
    examples=["/examples/test_image1.jpg"],
    title="Vietnam Traffic Sign Detection"
)

iface.launch()
