import numpy as np
from settings import * # Importer vos constantes

def load_data():
    """
    Fonction à implémenter pour charger votre dataset.
    Doit retourner (X_train, y_train), (X_test, y_test)
    - X_train, X_test doivent être de shape (N, 64, 64, 3) et normalisés (ex: / 255.0)
    - y_train, y_test doivent être en one-hot encoding (N, 5)
    """
    print("Chargement des données (placeholder)...")
    # Exemple de données factices
    N_train, N_test = 1000, 200
    X_train = np.random.rand(N_train, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS).astype(np.float32)
    X_test = np.random.rand(N_test, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS).astype(np.float32)
    
    y_train = np.random.randint(0, NUM_CLASSES, N_train)
    y_test = np.random.randint(0, NUM_CLASSES, N_test)
    
    # Conversion en one-hot
    from tensorflow.keras.utils import to_categorical
    y_train = to_categorical(y_train, NUM_CLASSES)
    y_test = to_categorical(y_test, NUM_CLASSES)
    
    return (X_train, y_train), (X_test, y_test)