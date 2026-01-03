# 1_train_float.py
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Importer vos fonctions et constantes
from settings import *
from data_loader import load_data

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

def build_pro_model():
    model = Sequential(name="cnn_float_pro")
    model.add(Input(shape=INPUT_SHAPE))
    
    # Bloc 1
    model.add(Conv2D(16, (3, 3), padding='same', use_bias=False)) # bias=False car BatchNormalisation le gère
    model.add(BatchNormalization()) # <--- MAGIE ICI
    model.add(tf.keras.layers.Activation('relu'))
    model.add(MaxPooling2D((2, 2))) 
    
    # Bloc 2
    model.add(Conv2D(32, (3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization()) # <--- MAGIE ICI
    model.add(tf.keras.layers.Activation('relu'))
    model.add(MaxPooling2D((2, 2))) 
    
    # Bloc 3
    model.add(Conv2D(64, (3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization()) # <--- MAGIE ICI
    model.add(tf.keras.layers.Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    
    # Bloc 4 (On garde pour le 64px)
    model.add(Conv2D(64, (3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Flatten())
    
    model.add(Dense(64, use_bias=False))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    
    model.add(Dropout(0.4)) # Un peu plus de dropout pour forcer l'apprentissage robuste
    model.add(Dense(NUM_CLASSES, activation='softmax', name='output_softmax'))
    
    return model

if __name__ == "__main__":
    # ... (Chargement données et DataAugmentation comme avant) ...
    (X_train, y_train), (X_test, y_test) = load_data()
    
    # Reprends ton ImageDataGenerator existant ici
    datagen = ImageDataGenerator(
        rotation_range=20,      # Un peu plus de rotation (20°)
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False 
    )
    datagen.fit(X_train)

    model = build_pro_model() # Utilise le nouveau modèle
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # --- CALLBACKS INTELLIGENTS ---
    # 1. Sauvegarde uniquement si le score de validation s'améliore
    checkpoint = ModelCheckpoint(MODEL_FLOAT_PATH, 
                                 monitor='val_accuracy', 
                                 verbose=1, 
                                 save_best_only=True, 
                                 mode='max')
                                 
    # 2. Réduit la vitesse d'apprentissage si on stagne (très efficace pour les derniers %)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=5, min_lr=0.00001, verbose=1)

    print("\n--- Entraînement PRO ---")
    model.fit(datagen.flow(X_train, y_train, batch_size=32),
              epochs=50, # On laisse le temps !
              validation_data=(X_test, y_test),
              callbacks=[checkpoint, reduce_lr]) # On ajoute les callbacks

    # On recharge le MEILLEUR modèle pour l'évaluation finale (pas le dernier)
    print("Chargement du meilleur modèle sauvegardé...")
    best_model = keras.models.load_model(MODEL_FLOAT_PATH)
    loss, acc = best_model.evaluate(X_test, y_test)
    print(f"Précision FINALE (Best): {acc*100:.2f}%")