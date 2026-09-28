"""
Data loading and preprocessing utilities.
Handles reading raw images, resizing, normalizing, and caching processed numpy arrays.
"""

import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Local import from src.settings
from src.settings import RAW_PATH, PROCESSED_PATH, CLASSES, NUM_CLASSES, IMG_WIDTH, IMG_HEIGHT

def process_raw_data():
    """
    Reads RAW images, resizes, normalizes, and returns numpy arrays.
    """
    images = []
    labels = []

    print(f"Processing raw data from: {RAW_PATH}")

    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"RAW directory {RAW_PATH} not found.")

    for label_index, category_name in enumerate(CLASSES):
        folder_path = os.path.join(RAW_PATH, category_name)
        
        if not os.path.exists(folder_path):
            print(f"Warning: Folder '{category_name}' missing in raw directory.")
            continue
            
        print(f" -> Processing '{category_name}'...")
        
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            
            if img is None: 
                continue 

            # Preprocessing: BGR -> RGB and Resize
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            
            images.append(img)
            labels.append(label_index)

    X = np.array(images, dtype='float32')
    y = np.array(labels, dtype='int')

    # Normalization
    X = X / 255.0
    
    # One-hot encoding
    y = to_categorical(y, num_classes=NUM_CLASSES)

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def load_data(force_reprocess=False):
    """
    Loads data from 'processed' cache. If not found or force_reprocess is True, processes 'raw' data.
    """
    files = {
        "X_train": os.path.join(PROCESSED_PATH, "X_train.npy"),
        "y_train": os.path.join(PROCESSED_PATH, "y_train.npy"),
        "X_test":  os.path.join(PROCESSED_PATH, "X_test.npy"),
        "y_test":  os.path.join(PROCESSED_PATH, "y_test.npy")
    }

    # Check if cached files exist
    data_exists = all(os.path.exists(f) for f in files.values())

    if data_exists and not force_reprocess:
        print("Loading processed data from cache...")
        X_train = np.load(files["X_train"])
        y_train = np.load(files["y_train"])
        X_test  = np.load(files["X_test"])
        y_test  = np.load(files["y_test"])
    else:
        print("No processed data found (or force_reprocess=True). Generating now...")
        
        # Heavy processing
        X_train, X_test, y_train, y_test = process_raw_data()
        
        # Create processed folder if it doesn't exist
        os.makedirs(PROCESSED_PATH, exist_ok=True)
            
        # Cache for next time
        np.save(files["X_train"], X_train)
        np.save(files["y_train"], y_train)
        np.save(files["X_test"], X_test)
        np.save(files["y_test"], y_test)
        print(f"Data saved to {PROCESSED_PATH}")

    print(f"Data ready - Train: {X_train.shape}, Test: {X_test.shape}")
    return (X_train, y_train), (X_test, y_test)

if __name__ == "__main__":
    # Test execution: force regeneration
    load_data(force_reprocess=True)