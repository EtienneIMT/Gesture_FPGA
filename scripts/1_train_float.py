# 1_train_float.py
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam

# Importer vos fonctions et constantes
from settings import *
from data_loader import load_data # Fichier fictif 'data_loader.py' contenant la fonction load_data

def build_float_model():
    model = Sequential(name="cnn_float")
    
    model.add(Input(shape=INPUT_SHAPE))
    
    # Bloc 1
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same', name='conv1'))
    model.add(MaxPooling2D((2, 2), name='pool1')) # -> 32x32

    # Bloc 2
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same', name='conv2'))
    model.add(MaxPooling2D((2, 2), name='pool2')) # -> 16x16
    
    # Bloc 3
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='conv3'))
    model.add(MaxPooling2D((2, 2), name='pool3')) # -> 8x8

    # Tête de classification
    model.add(Flatten(name='flatten'))
    model.add(Dense(32, activation='relu', name='fc1'))
    model.add(Dropout(0.5, name='dropout'))
    model.add(Dense(NUM_CLASSES, activation='softmax', name='output_softmax'))
    
    return model

# --- Main script ---
if __name__ == "__main__":
    # 1. Charger les données
    (X_train, y_train), (X_test, y_test) = load_data()

    # 2. Construire le modèle
    model = build_float_model()
    model.summary()

    # 3. Compiler
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 4. Entraîner
    print("\n--- Entraînement du modèle flottant ---")
    model.fit(X_train, y_train,
              batch_size=32,
              epochs=20,
              validation_data=(X_test, y_test),
              shuffle=True)

    # 5. Évaluer
    print("\n--- Évaluation du modèle flottant ---")
    loss, acc = model.evaluate(X_test, y_test)
    print(f"Précision (float): {acc*100:.2f}%")

    # 6. Sauvegarder le modèle
    model.save(MODEL_FLOAT_PATH)
    print(f"Modèle flottant sauvegardé dans {MODEL_FLOAT_PATH}")