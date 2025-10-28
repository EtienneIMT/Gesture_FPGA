#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
infer_test.py - Run inference on trained gesture recognition model.
Usage:
  python scripts/infer_test.py --model models/cnn_gesture_v1.pt --source cam
  python scripts/infer_test.py --model models/cnn_gesture_v1.pt --source data/test/open_hand.jpg
"""

import argparse
import torch
import cv2
import time
from torchvision import transforms
from PIL import Image
import numpy as np

from models.cnn_gesture_v1 import GestureNet

# -------------------------------
# Argument parsing
# -------------------------------
parser = argparse.ArgumentParser(description="Gesture recognition inference script.")
parser.add_argument("--model", type=str, required=True, help="Path to the trained model (.pt or .onnx).")
parser.add_argument("--source", type=str, default="cam", help="Source: 'cam' for webcam or path to image file.")
parser.add_argument("--labels", type=str, default="data/labels.json", help="Path to label mapping (optional).")
args = parser.parse_args()

# -------------------------------
# Device and model loading
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Optional: load label names
import json
import os

# --- Determine number of classes ---
num_classes = 24 # Default for SignMNIST (A-Y, excluding J, Z)
label_map = None # Initialize label_map

if os.path.exists(args.labels):
    try:
        with open(args.labels, "r") as f:
             label_map = json.load(f)
             # Verify consistency (optional but good practice)
             if len(label_map) != num_classes:
                 print(f"Warning: Labels file has {len(label_map)} entries, but expected {num_classes}.")
             print(f"Loaded {len(label_map)} labels from {args.labels}")
    except Exception as e:
        print(f"Warning: Could not read labels file {args.labels}: {e}. Using default SignMNIST labels.")
        label_map = None # Ensure fallback is used if file loading fails

# --- Create Fallback Label Map IF NOT loaded from file ---
if label_map is None:
    print("Using default SignMNIST labels (A-Y, excluding J/Z).")
    # Labels 0-8 are A-I
    label_map = {str(i): chr(ord('A') + i) for i in range(9)}
    # Labels 9-23 map to K-Y (skip J)
    for i in range(9, 24):
        label_map[str(i)] = chr(ord('A') + i + 1) # +1 to skip 'J'
    # num_classes should ideally be derived dynamically earlier if possible,
    # but for fallback, 24 is assumed for SignMNIST.
    num_classes = 24

# Load model
model = GestureNet(num_classes=num_classes)

try:
    state_dict = torch.load(args.model, map_location=device)
    model.load_state_dict(state_dict) # <--- LOAD WEIGHTS INTO MODEL
    print(f"Successfully loaded weights from {args.model}")
except Exception as e:
    print(f"Error loading model weights: {e}")
    exit(1)

model.eval()
model.to(device)

# -------------------------------
# Transform definition (same as training)
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# -------------------------------
# Inference functions
# -------------------------------
def predict_image(img_pil):
    img_pil_gray = img_pil.convert("L")
    # Apply transform but DON'T add batch dimension yet
    img_tensor_transformed = transform(img_pil_gray) 
    
    # Add batch dimension for inference
    img_tensor_batch = img_tensor_transformed.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor_batch)
        _, predicted = torch.max(outputs, 1)
    label = label_map.get(str(predicted.item()), str(predicted.item()))
    return label, img_tensor_transformed

# --- Function to convert tensor back to displayable image ---
def tensor_to_cv2_image(tensor):
    """Converts a transformed PyTorch tensor back to an OpenCV image."""
    # 1. Move tensor to CPU and detach from gradient graph
    img = tensor.cpu().detach()
    
    # 2. Reverse normalization: (tensor * std) + mean
    # Our normalization was Normalize((0.5,), (0.5,)), so mean=0.5, std=0.5
    img = img * 0.5 + 0.5 
    
    # 3. Convert to NumPy array
    img_np = img.numpy()
    
    # 4. Remove channel dimension if it's 1 (for grayscale)
    if img_np.shape[0] == 1:
        img_np = np.squeeze(img_np, axis=0) 
        
    # 5. Scale to 0-255 and convert to uint8
    img_display = (img_np * 255).astype(np.uint8)
    
    return img_display

def predict_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera.")
        return

    print("🎥 Camera started. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame.")
            break

        # Convert frame for inference
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        label, transformed_tensor = predict_image(img_pil)

        # Convert transformed tensor to displayable image
        transformed_img_display = tensor_to_cv2_image(transformed_tensor)

        # Display result
        cv2.putText(frame, f"Gesture: {label}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.imshow("Camera Feed", frame)

        # Show transformed image in a separate window
        cv2.imshow("Transformed Input", transformed_img_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# -------------------------------
# Run inference
# -------------------------------
if args.source == "cam":
    predict_camera()
else:
    if not os.path.exists(args.source):
        raise FileNotFoundError(f"Image file not found: {args.source}")
    img_pil = Image.open(args.source).convert("RGB")
    label = predict_image(img_pil)
    print(f"Predicted gesture: {label}")
