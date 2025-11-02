# settings.py
# Constantes du projet
NUM_CLASSES = 5
IMG_HEIGHT = 64
IMG_WIDTH = 64
IMG_CHANNELS = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# Chemin vers les modèles
MODEL_FLOAT_PATH = 'models/gesture_cnn_float.h5'
MODEL_QAT_PATH = 'models/gesture_cnn_quantized.h5'

# Répertoire du projet HLS
HLS_PROJECT_PATH = 'hls_gesture_project'