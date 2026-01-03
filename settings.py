# settings.py

import os

# --- CHEMINS ---
# Chemin de base du projet
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dossier Data global
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Dossier des images originales (à créer manuellement et remplir)
RAW_PATH = os.path.join(DATA_DIR, 'clean') # Pointer vers le dossier nettoyé

# Dossier des données Numpy prêtes (sera créé automatiquement)
PROCESSED_PATH = os.path.join(DATA_DIR, 'processed')

# Constantes du projet
CLASSES = ["fist", "grip", "little_finger", "peace", "thumb_index"]
NUM_CLASSES = len(CLASSES)
IMG_HEIGHT = 64
IMG_WIDTH = 64
IMG_CHANNELS = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# Chemin vers les modèles
MODEL_FLOAT_PATH = 'models/gesture_cnn_float.h5'
MODEL_QAT_PATH = 'models/gesture_cnn_quantized.h5'

# Répertoire du projet HLS
HLS_PROJECT_PATH = 'hls_gesture_project'