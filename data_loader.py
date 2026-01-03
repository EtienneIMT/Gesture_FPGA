import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from settings import *

def process_raw_data():
    """
    Lit les images RAW, redimensionne, normalise et retourne les tableaux numpy.
    """
    images = []
    labels = []

    print(f"Traitement des données brutes depuis : {RAW_PATH}")

    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"Le dossier RAW {RAW_PATH} est introuvable.")

    for label_index, category_name in enumerate(CLASSES):
        folder_path = os.path.join(RAW_PATH, category_name)
        
        if not os.path.exists(folder_path):
            print(f"Attention : Dossier '{category_name}' manquant dans raw.")
            continue
            
        print(f" -> Traitement '{category_name}'...")
        
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            
            if img is None: continue 

            # Traitement : BGR -> RGB et Resize
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            
            images.append(img)
            labels.append(label_index)

    X = np.array(images, dtype='float32')
    y = np.array(labels, dtype='int')

    # Normalisation
    X = X / 255.0
    # One-hot encoding
    y = to_categorical(y, num_classes=NUM_CLASSES)

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def load_data(force_reprocess=False):
    """
    Charge les données depuis 'processed'. Si elles n'existent pas, traite 'raw'.
    """
    # Chemins des fichiers sauvegardés
    files = {
        "X_train": os.path.join(PROCESSED_PATH, "X_train.npy"),
        "y_train": os.path.join(PROCESSED_PATH, "y_train.npy"),
        "X_test":  os.path.join(PROCESSED_PATH, "X_test.npy"),
        "y_test":  os.path.join(PROCESSED_PATH, "y_test.npy")
    }

    # Vérifier si les fichiers existent déjà
    data_exists = all(os.path.exists(f) for f in files.values())

    if data_exists and not force_reprocess:
        print("Chargement des données traitées depuis le cache...")
        X_train = np.load(files["X_train"])
        y_train = np.load(files["y_train"])
        X_test  = np.load(files["X_test"])
        y_test  = np.load(files["y_test"])
    else:
        print("Aucune donnée traitée trouvée (ou force_reprocess=True). Génération en cours...")
        
        # On lance le traitement lourd
        X_train, X_test, y_train, y_test = process_raw_data()
        
        # On crée le dossier processed s'il n'existe pas
        if not os.path.exists(PROCESSED_PATH):
            os.makedirs(PROCESSED_PATH)
            
        # On sauvegarde pour la prochaine fois
        np.save(files["X_train"], X_train)
        np.save(files["y_train"], y_train)
        np.save(files["X_test"], X_test)
        np.save(files["y_test"], y_test)
        print(f"Données sauvegardées dans {PROCESSED_PATH}")

    print(f"Données prêtes : Train {X_train.shape}, Test {X_test.shape}")
    return (X_train, y_train), (X_test, y_test)

if __name__ == "__main__":
    # Test : force la régénération
    load_data(force_reprocess=True)