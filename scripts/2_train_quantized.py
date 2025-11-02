# 2_train_quantized.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from keras.models import Model, load_model
from keras.layers import Input, MaxPooling2D, Flatten, Dropout

# Importer QKeras
import qkeras
from qkeras.quantizers import quantized_bits, quantized_relu
from qkeras import QConv2D, QDense, QActivation

# Importer vos fonctions et constantes
from settings import *
from data_loader import load_data

# Définir notre quantificateur. INT8 = 8 bits, 0 bits entiers (pour poids entre -1 et 1)
# Pour INT4, utilisez quantized_bits(4, 0, alpha=1)
QUANT_8BIT = quantized_bits(bits=8, integer=0, alpha=1)
RELU_8BIT = quantized_relu(bits=8)

def build_quantized_model():
    """ Construit le modèle QKeras. Notez l'absence de softmax. """
    
    inp = Input(shape=INPUT_SHAPE, name='input_1') # Doit correspondre au nom d'input Keras
    
    # Bloc 1
    x = QConv2D(8, (3, 3), 
                kernel_quantizer=QUANT_8BIT, bias_quantizer=QUANT_8BIT,
                padding='same', name='conv1')(inp)
    x = QActivation(RELU_8BIT, name='relu1')(x)
    x = MaxPooling2D((2, 2), name='pool1')(x)

    # Bloc 2
    x = QConv2D(16, (3, 3), 
                kernel_quantizer=QUANT_8BIT, bias_quantizer=QUANT_8BIT,
                padding='same', name='conv2')(x)
    x = QActivation(RELU_8BIT, name='relu2')(x)
    x = MaxPooling2D((2, 2), name='pool2')(x)
    
    # Bloc 3
    x = QConv2D(32, (3, 3), 
                kernel_quantizer=QUANT_8BIT, bias_quantizer=QUANT_8BIT,
                padding='same', name='conv3')(x)
    x = QActivation(RELU_8BIT, name='relu3')(x)
    x = MaxPooling2D((2, 2), name='pool3')(x)

    # Tête de classification
    x = Flatten(name='flatten')(x)
    x = QDense(32, kernel_quantizer=QUANT_8BIT, bias_quantizer=QUANT_8BIT, name='fc1')(x)
    x = QActivation(RELU_8BIT, name='relu_fc1')(x)
    x = Dropout(0.5, name='dropout')(x) # Le Dropout est ignoré lors de l'inférence/conversion HLS
    
    # Couche de sortie (logits) - SANS ACTIVATION SOFTMAX
    out = QDense(NUM_CLASSES, 
                 kernel_quantizer=QUANT_8BIT, bias_quantizer=QUANT_8BIT,
                 name='output_logits')(x)
    
    return Model(inputs=inp, outputs=out, name="cnn_quantized")

# --- Main script ---
if __name__ == "__main__":
    # 1. Charger les données
    (X_train, y_train), (X_test, y_test) = load_data()

    # 2. Construire le modèle QAT
    model_q = build_quantized_model()
    model_q.summary()
    
    # 3. (Optionnel mais recommandé) Charger les poids du modèle flottant
    try:
        model_q.load_weights(MODEL_FLOAT_PATH, by_name=True, skip_mismatch=True)
        print("Poids du modèle flottant chargés avec succès.")
    except Exception as e:
        print(f"Erreur lors du chargement des poids flottants (ignorer si premier entraînement): {e}")

    # 4. Compiler
    # IMPORTANT: Nous utilisons la 'loss' AVEC logits, car notre modèle ne sort pas de softmax.
    model_q.compile(optimizer=Adam(learning_rate=0.0005), # Taux d'apprentissage plus faible pour le fine-tuning
                    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

    # 5. Entraîner (QAT)
    print("\n--- Entraînement du modèle quantisé (QAT) ---")
    model_q.fit(X_train, y_train,
                batch_size=32,
                epochs=15, # Moins d'époques pour le fine-tuning
                validation_data=(X_test, y_test),
                shuffle=True)

    # 6. Évaluer
    print("\n--- Évaluation du modèle quantisé ---")
    loss, acc = model_q.evaluate(X_test, y_test)
    print(f"Précision (quantisée): {acc*100:.2f}%")

    # 7. Sauvegarder le modèle QAT
    model_q.save(MODEL_QAT_PATH)
    print(f"Modèle quantisé sauvegardé dans {MODEL_QAT_PATH}")