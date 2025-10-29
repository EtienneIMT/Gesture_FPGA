#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
infer_test.py - Run inference on trained gesture recognition model.
Usage:
  python scripts/infer_test.py --model models/py_models/cnn_gesture_v1.pt --source cam
  python scripts/infer_test.py --model models/py_models/cnn_gesture_v1.pt --source data/test/open_hand.jpg
"""

import argparse
import torch
import cv2
import time
from torchvision import transforms
from PIL import Image
import numpy as np
import mediapipe as mp

# Optional: load label names
import json
import os

from models.py_models.cnn_gesture_v1 import GestureNet

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


# --- Determine number of classes ---
num_classes = 5 # Default for SignMNIST (A-Y, excluding J, Z)
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
    print("Using default 5-class labels (A, I, L, O, V).")
    # Define the specific mapping for your chosen classes
    label_map = {
        '0': 'A',
        '1': 'L',
        '2': 'O',
        '3': 'V',
        '4': 'I'
    }
    num_classes = 5

print(f"Model configured for {num_classes} classes.")

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
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# -------------------------------
# Initialize CLAHE object
# Parameters: clipLimit (contrast limit), tileGridSize (size of subregions)
# Experiment with these values. Common values are clipLimit=2.0-4.0, tileGridSize=(8,8)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))

# -------------------------------
# Initialize MediaPipe Hands
# -------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,      # Traite un flux vidéo
    max_num_hands=1,             # Détecte une seule main
    min_detection_confidence=0.6, # Seuil de confiance pour détecter une main
    min_tracking_confidence=0.5) # Seuil pour suivre la main détectée
mp_drawing = mp.solutions.drawing_utils # Pour dessiner les points/boîtes

# -------------------------------
# Inference functions
# -------------------------------
def predict_image(img_pil_gray):
    """Performs inference on a single grayscale PIL image."""
    img_tensor_transformed = transform(img_pil_gray)
    img_tensor_batch = img_tensor_transformed.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor_batch)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        
    # Convertit l'index prédit (un tenseur/nombre) en chaîne de caractères AVANT la recherche
    predicted_label_str = str(predicted_idx.item()) 
    
    # Utilise .get() pour chercher la lettre; retourne l'index si non trouvé
    label = label_map.get(predicted_label_str, f"Unknown({predicted_label_str})")
    
    return label, confidence.item(), img_tensor_transformed

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

    print("🎥 Camera started. Press 'CTRL + C' to quit.")
    start_time = time.time()
    frame_count = 0
    label, confidence = "N/A", 0.0 # Initial values

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame.")
            break

        frame_count += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # MediaPipe a besoin de RGB
        frame_height, frame_width, _ = frame.shape
        
        # --- Détection MediaPipe ---
        results = hands.process(frame_rgb)
        
        hand_crop_for_inference = None # Image recadrée à envoyer au CNN
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # --- Calculer la Bounding Box ---
                x_coords = [landmark.x for landmark in hand_landmarks.landmark]
                y_coords = [landmark.y for landmark in hand_landmarks.landmark]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                # Convertir coordonnées normalisées en pixels (avec marge)
                padding = 0.1 # Ajoute 10% de marge
                box_x_min = max(0, int((x_min - padding) * frame_width))
                box_y_min = max(0, int((y_min - padding) * frame_height))
                box_x_max = min(frame_width, int((x_max + padding) * frame_width))
                box_y_max = min(frame_height, int((y_max + padding) * frame_height))

                # Dessiner la boite sur l'image originale (optionnel)
                cv2.rectangle(frame, (box_x_min, box_y_min), (box_x_max, box_y_max), (0, 255, 0), 2)
                # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS) # Dessine les points clés

                # --- Recadrer l'image ---
                # Recadre l'image N&B prétraitée avec CLAHE
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                clahe_frame = clahe.apply(gray_frame)
                
                # S'assurer que les coordonnées sont valides
                if box_y_min < box_y_max and box_x_min < box_x_max:
                     hand_crop_for_inference = clahe_frame[box_y_min:box_y_max, box_x_min:box_x_max] # Recadrer la version CLAHE
                
                break # Traite une seule main

        # --- Inférence sur l'image recadrée (si une main est détectée) ---
        transformed_img_display = np.zeros((64, 64), dtype=np.uint8) # Image noire si pas de main
        if hand_crop_for_inference is not None and hand_crop_for_inference.size > 0:
            try:
                # Convertir le crop NumPy en PIL pour les transformations
                hand_crop_pil = Image.fromarray(hand_crop_for_inference) 
                
                # Inférence
                label, confidence, transformed_tensor = predict_image(hand_crop_pil)
                
                # Préparer la visualisation du tenseur transformé
                transformed_img_display = tensor_to_cv2_image(transformed_tensor)
            except Exception as e:
                print(f"Erreur pendant l'inférence sur le crop: {e}")
                label, confidence = "Error", 0.0
        else:
            label, confidence = "No Hand", 0.0 # Pas de main détectée

        # Calculate FPS
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        # Display results
        display_text = f"Gesture: {label} ({confidence*100:.1f}%) FPS: {fps:.1f}"
        cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        cv2.imshow("Camera Feed + Detection", frame) 
        cv2.imshow("Input to CNN (Cropped & Transformed)", transformed_img_display) 

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    hands.close() # Libérer les ressources MediaPipe
    cv2.destroyAllWindows()
    print(f"Average FPS: {frame_count / (time.time() - start_time):.1f}")

# -------------------------------
# Run inference
# -------------------------------
if args.source == "cam":
    predict_camera()
else:
    if not os.path.exists(args.source):
        print(f"❌ Image file not found: {args.source}")
        exit(1)
    try:
        print("Inference on single image not yet updated for MediaPipe cropping.")
        img_pil = Image.open(args.source)
        # Convert to grayscale NumPy array for CLAHE
        img_np_rgb = np.array(img_pil)
        img_np_gray = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2GRAY)
        
        # Apply CLAHE
        clahe_processed_img = clahe.apply(img_np_gray)

        label, confidence, transformed_tensor = predict_image(clahe_processed_img) 
        print(f"Predicted gesture: {label} (Confidence: {confidence*100:.1f}%)")
        
        transformed_img_display = tensor_to_cv2_image(transformed_tensor)
        
        # Original (colored if was) and CLAHE versions
        cv2.imshow("Original Image", img_np_rgb) 
        cv2.imshow("CLAHE Preprocessed", clahe_processed_img) # <--- NEW WINDOW
        cv2.imshow("Transformed Input (to Model)", transformed_img_display)
        print("Press any key in an image window to exit.")
        cv2.waitKey(0) 
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error processing image file {args.source}: {e}")
