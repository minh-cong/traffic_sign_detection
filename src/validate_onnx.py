import onnxruntime as ort
import numpy as np

# Load model ONNX
sess = ort.InferenceSession("best.onnx")
input_name = sess.get_inputs()[0].name

# Tạo input giả
fake_input = np.random.randn(1, 3, 640, 640).astype(np.float32)

# Chạy inference
outputs = sess.run(None, {input_name: fake_input})
print("ONNX model works!" if outputs else "Error!")
