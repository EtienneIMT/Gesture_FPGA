# scripts/export_onnx.py
import torch
from models.simple_cnn import SimpleCNN

model = SimpleCNN(n_classes=4)
model.eval()
dummy = torch.randn(1, 3, 64, 64)
torch.onnx.export(
    model,
    dummy,
    "onnx/gesture_float.onnx",
    opset_version=18,
    input_names=["input"],
    output_names=["output"],
)
print("ONNX saved: onnx/gesture_float.onnx")
