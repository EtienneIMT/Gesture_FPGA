# 2_train_quantized.py (Version Optimisée)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, MaxPooling2D, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau # <--- AJOUT IMPORTANT
import qkeras
from qkeras import QConv2D, QDense, QActivation

from settings import *
from data_loader import load_data

def build_quantized_model():
    # On garde ta configuration 6 bits qui est très bien pour le FPGA
    kwargs = {
        'kernel_quantizer': 'quantized_bits(6,0,alpha=1)',
        'bias_quantizer': 'quantized_bits(16,6,alpha=1)'
    }

    model = Sequential(name="cnn_quantized")
    model.add(Input(shape=INPUT_SHAPE))
    
    # Bloc 1
    model.add(QConv2D(8, (3, 3), padding='same', name='conv1', **kwargs))
    model.add(QActivation('quantized_relu(6,0)', name='act1'))
    model.add(MaxPooling2D((2, 2), name='pool1'))

    # Bloc 2
    model.add(QConv2D(16, (3, 3), padding='same', name='conv2', **kwargs))
    model.add(QActivation('quantized_relu(6,0)', name='act2'))
    model.add(MaxPooling2D((2, 2), name='pool2'))
    
    # Bloc 3
    model.add(QConv2D(32, (3, 3), padding='same', name='conv3', **kwargs))
    model.add(QActivation('quantized_relu(6,0)', name='act3'))
    model.add(MaxPooling2D((2, 2), name='pool3'))

    # Classification
    model.add(Flatten(name='flatten'))
    model.add(QDense(32, name='fc1', **kwargs))
    model.add(QActivation('quantized_relu(6,0)', name='act4'))
    model.add(Dropout(0.5, name='dropout'))
    
    model.add(QDense(NUM_CLASSES, name='output_dense', **kwargs))
    model.add(keras.layers.Activation('softmax', name='output_softmax'))
    
    return model

if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = load_data()

    qmodel = build_quantized_model()
    
    # Tentative de chargement des poids flottants
    print(f"Chargement des poids depuis : {MODEL_FLOAT_PATH}")
    try:
        # On charge les poids "by_name" pour être plus robuste
        float_model = keras.models.load_model(MODEL_FLOAT_PATH)
        for layer in qmodel.layers:
            if isinstance(layer, (QConv2D, QDense)):
                try:
                    # Trouve la couche correspondante dans le modèle float
                    float_layer = float_model.get_layer(layer.name)
                    layer.set_weights(float_layer.get_weights())
                    print(f" -> Poids transférés pour {layer.name}")
                except:
                    print(f" -> Pas de poids trouvés pour {layer.name} (normal si architecture diffère)")
    except Exception as e:
        print(f"Erreur chargement poids: {e}")
        print("Démarrage from scratch.")

    qmodel.compile(optimizer=Adam(learning_rate=0.0005),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

    # --- CALLBACKS (LA CLÉ DU SUCCÈS) ---
    checkpoint = ModelCheckpoint(MODEL_QAT_PATH, 
                                 monitor='val_accuracy', 
                                 verbose=1, 
                                 save_best_only=True, 
                                 mode='max')
    
    # Réduire la vitesse si on bloque
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    print("\n--- Entraînement Quantifié Long (40 époques) ---")
    qmodel.fit(X_train, y_train,
               batch_size=32,
               epochs=40, # On augmente la durée !
               validation_data=(X_test, y_test),
               callbacks=[checkpoint, reduce_lr])

    print("\n--- Résultat Final (Meilleur Modèle) ---")
    # On recharge le meilleur
    best_qmodel = keras.models.load_model(MODEL_QAT_PATH, 
                                          custom_objects={'QConv2D': QConv2D, 'QActivation': QActivation, 'QDense': QDense})
    loss, acc = best_qmodel.evaluate(X_test, y_test)
    print(f"Précision Quantifiée FINALE: {acc*100:.2f}%")