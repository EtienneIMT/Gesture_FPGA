"""
Clean and preprocess raw gesture images using MediaPipe for hand tracking.
Extracts hand bounding boxes, applies padding, and resizes to the target input size.
"""

import sys
import os
import cv2
import mediapipe as mp
import numpy as np

# Ensure the root directory is on the path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.settings import DATA_DIR, RAW_PATH, IMG_HEIGHT

# Configuration
RAW_DIR = os.path.join(DATA_DIR, "raw")
CLEAN_DIR = RAW_PATH # Using the clean path from settings
TARGET_SIZE = IMG_HEIGHT
PADDING = 20

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True, 
    max_num_hands=1, 
    min_detection_confidence=0.5
)

def process_dataset():
    if not os.path.exists(CLEAN_DIR):
        os.makedirs(CLEAN_DIR)

    # Process each gesture class directory
    for class_name in os.listdir(RAW_DIR):
        raw_class_path = os.path.join(RAW_DIR, class_name)
        clean_class_path = os.path.join(CLEAN_DIR, class_name)
        
        if not os.path.isdir(raw_class_path): 
            continue
        if not os.path.exists(clean_class_path): 
            os.makedirs(clean_class_path)

        print(f"Processing class: {class_name}...")
        
        count = 0
        for img_name in os.listdir(raw_class_path):
            img_path = os.path.join(raw_class_path, img_name)
            image = cv2.imread(img_path)
            if image is None: 
                continue

            # MediaPipe expects RGB images
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                # Process the first detected hand
                hand_landmarks = results.multi_hand_landmarks[0]
                h, w, c = image.shape
                
                # Compute Bounding Box
                x_min, y_min = w, h
                x_max, y_max = 0, 0
                
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    if x < x_min: x_min = x
                    if x > x_max: x_max = x
                    if y < y_min: y_min = y
                    if y > y_max: y_max = y
                
                # Apply padding
                x_min = max(0, x_min - PADDING)
                y_min = max(0, y_min - PADDING)
                x_max = min(w, x_max + PADDING)
                y_max = min(h, y_max + PADDING)
                
                # Make the bounding box square
                box_w = x_max - x_min
                box_h = y_max - y_min
                
                if box_w > box_h:
                    diff = (box_w - box_h) // 2
                    y_min = max(0, y_min - diff)
                    y_max = min(h, y_max + diff)
                else:
                    diff = (box_h - box_w) // 2
                    x_min = max(0, x_min - diff)
                    x_max = min(w, x_max + diff)

                # Crop and Resize
                crop = image[y_min:y_max, x_min:x_max]
                if crop.size != 0:
                    crop_resized = cv2.resize(crop, (TARGET_SIZE, TARGET_SIZE))
                    
                    # Save cropped image
                    save_path = os.path.join(clean_class_path, img_name)
                    cv2.imwrite(save_path, crop_resized)
                    count += 1
        
        print(f" -> Saved {count} cleaned images.")

if __name__ == "__main__":
    process_dataset()