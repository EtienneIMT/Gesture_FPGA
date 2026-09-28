"""
Preprocess the SignMNIST dataset from CSV to image folders.
Filters target classes and splits into train/validation/test sets.
"""

import os
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import sys

# Ensure the root directory is on the path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.settings import DATA_DIR, CLASSES

# --- Configuration ---
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
TRAIN_CSV = os.path.join(RAW_DATA_DIR, "sign_mnist_train.csv")
TEST_CSV = os.path.join(RAW_DATA_DIR, "sign_mnist_test.csv")

# Output directories for ImageFolder structure
PROCESSED_DIR = os.path.join(DATA_DIR, "processed_images")
TRAIN_DIR = os.path.join(PROCESSED_DIR, "train")
VAL_DIR = os.path.join(PROCESSED_DIR, "val")
TEST_DIR = os.path.join(PROCESSED_DIR, "test")

# Validation split percentage (e.g., 20% of training data for validation)
VAL_SPLIT_RATIO = 0.2 
RANDOM_SEED = 42

# We map our 5 classes to specific SignMNIST letters (A=0, I=8, L=11, O=14, V=21)
TARGET_CLASSES = [0, 8, 11, 14, 21] 
TARGET_CLASS_STR = [str(c) for c in TARGET_CLASSES]

# --- Function to process CSV and save images ---
def csv_to_image_folders(csv_path, base_output_dir, target_labels):
    """Reads SignMNIST CSV, filters for target labels, and saves images."""
    print(f"Processing {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        print("Please ensure the CSV files are in the RAW_DATA_DIR.")
        return None, None
    
    df_filtered = df[df['label'].isin(target_labels)]
    print(f"  Filtered down to {len(df_filtered)} samples for classes {target_labels}")

    labels = df_filtered['label'].values
    pixels = df_filtered.drop('label', axis=1).values 

    os.makedirs(base_output_dir, exist_ok=True)

    image_paths = []
    image_labels = []

    num_images = len(labels)
    for i in range(num_images):
        label = labels[i]
        img_array = pixels[i].reshape(28, 28).astype(np.uint8) 
        img = Image.fromarray(img_array)

        class_dir = os.path.join(base_output_dir, str(label))
        os.makedirs(class_dir, exist_ok=True)
        
        img_filename = f"img_{i}.png"
        img_path = os.path.join(class_dir, img_filename)
        img.save(img_path)

        image_paths.append(img_path)
        image_labels.append(label)

        if (i + 1) % 1000 == 0:
            print(f"  Saved {i + 1}/{num_images} images...")

    print(f"Finished saving images to {base_output_dir}")
    return image_paths, image_labels

if __name__ == "__main__":
    if not os.path.isdir(RAW_DATA_DIR):
        print(f"Creating raw data directory: {RAW_DATA_DIR}")
        print(f"Please place sign_mnist_train.csv and sign_mnist_test.csv inside {RAW_DATA_DIR}")
        os.makedirs(RAW_DATA_DIR)
        if not (os.path.exists(TRAIN_CSV) and os.path.exists(TEST_CSV)):
             print("Exiting. Place CSV files in data/raw/ and rerun.")
             exit()

    print("\n--- Processing Test Set ---")
    test_paths, _ = csv_to_image_folders(TEST_CSV, TEST_DIR, TARGET_CLASSES)
    if test_paths is None:
        exit()

    print("\n--- Processing Training Set ---")
    TEMP_TRAIN_DIR = os.path.join(PROCESSED_DIR, "temp_train")
    train_val_paths, train_val_labels = csv_to_image_folders(TRAIN_CSV, TEMP_TRAIN_DIR, TARGET_CLASSES)
    if train_val_paths is None:
        exit()

    print("\n--- Splitting Training Data into Train/Validation ---")
    train_paths, val_paths, _, _ = train_test_split(
        train_val_paths, 
        train_val_labels, 
        test_size=VAL_SPLIT_RATIO, 
        random_state=RANDOM_SEED,
        stratify=train_val_labels
    )

    os.makedirs(VAL_DIR, exist_ok=True)
    print(f"Moving {len(val_paths)} files to {VAL_DIR}...")
    for file_path in val_paths:
        parts = file_path.split(os.sep)
        label = parts[-2]
        filename = parts[-1]
        
        dest_class_dir = os.path.join(VAL_DIR, label)
        os.makedirs(dest_class_dir, exist_ok=True)
        dest_path = os.path.join(dest_class_dir, filename)
        
        try:
            shutil.move(file_path, dest_path)
        except Exception as e:
            print(f"Error moving {file_path} to {dest_path}: {e}")

    print(f"Renaming {TEMP_TRAIN_DIR} to {TRAIN_DIR}...")
    try:
        if os.path.exists(TRAIN_DIR):
             shutil.rmtree(TRAIN_DIR)
        os.rename(TEMP_TRAIN_DIR, TRAIN_DIR)
    except Exception as e:
        print(f"Error renaming directory: {e}")

    print("\n--- Dataset Preparation Complete ---")
    print(f"Training images: {len(train_paths)} (in {TRAIN_DIR})")
    print(f"Validation images: {len(val_paths)} (in {VAL_DIR})")
    print(f"Test images: {len(test_paths)} (in {TEST_DIR})")