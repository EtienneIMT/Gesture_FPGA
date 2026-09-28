# src/settings.py

import os

# --- PATHS ---
# Base project directory (assuming this file is in src/)
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SRC_DIR, '..'))

# Data directories
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_PATH = os.path.join(DATA_DIR, 'clean') 
PROCESSED_PATH = os.path.join(DATA_DIR, 'processed')

# Model output directories
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_FLOAT_PATH = os.path.join(MODELS_DIR, 'gesture_cnn_float.h5')
MODEL_QAT_PATH = os.path.join(MODELS_DIR, 'gesture_cnn_quantized.h5')

# HLS Project directory
HLS_PROJECT_PATH = os.path.join(BASE_DIR, 'hls_gesture_project')

# --- PROJECT CONSTANTS ---
CLASSES = ["fist", "grip", "little_finger", "peace", "thumb_index"]
NUM_CLASSES = len(CLASSES)
IMG_HEIGHT = 64
IMG_WIDTH = 64
IMG_CHANNELS = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)