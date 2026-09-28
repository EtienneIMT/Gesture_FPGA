"""
Quantization-Aware Training (QAT) to prepare the CNN for FPGA deployment.
Converts the float32 architecture into a low-precision model (e.g., 6-bit weights/activations).
"""

import sys
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, MaxPooling2D, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import qkeras
from qkeras import QConv2D, QDense, QActivation

# Ensure the root directory is on the path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.settings import INPUT_SHAPE, NUM_CLASSES, MODEL_FLOAT_PATH, MODEL_QAT_PATH
from src.data_loader import load_data

def build_quantized_model():
    """Builds the QKeras equivalent of the baseline CNN using 6-bit quantization."""
    kwargs = {
        'kernel_quantizer': 'quantized_bits(6,0,alpha=1)',
        'bias_quantizer': 'quantized_bits(16,6,alpha=1)'
    }

    model = Sequential(name="cnn_quantized")
    model.add(Input(shape=INPUT_SHAPE))
    
    # Block 1
    model.add(QConv2D(8, (3, 3), padding='same', name='conv1', **kwargs))
    model.add(QActivation('quantized_relu(6,0)', name='act1'))
    model.add(MaxPooling2D((2, 2), name='pool1'))

    # Block 2
    model.add(QConv2D(16, (3, 3), padding='same', name='conv2', **kwargs))
    model.add(QActivation('quantized_relu(6,0)', name='act2'))
    model.add(MaxPooling2D((2, 2), name='pool2'))
    
    # Block 3
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
    print("Loading data...")
    (X_train, y_train), (X_test, y_test) = load_data()

    qmodel = build_quantized_model()
    
    # Attempt to load pretrained float weights as a starting point
    print(f"Loading pretrained weights from: {MODEL_FLOAT_PATH}")
    try:
        float_model = keras.models.load_model(MODEL_FLOAT_PATH)
        for layer in qmodel.layers:
            if isinstance(layer, (QConv2D, QDense)):
                try:
                    float_layer = float_model.get_layer(layer.name)
                    layer.set_weights(float_layer.get_weights())
                    print(f" -> Transferred weights for {layer.name}")
                except ValueError:
                    print(f" -> No weights transferred for {layer.name} (architecture differs)")
    except Exception as e:
        print(f"Warning: Could not load float weights ({e}). Training from scratch.")

    qmodel.compile(optimizer=Adam(learning_rate=0.0005),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

    # Callbacks
    checkpoint = ModelCheckpoint(MODEL_QAT_PATH, 
                                 monitor='val_accuracy', 
                                 verbose=1, 
                                 save_best_only=True, 
                                 mode='max')
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    print("\n--- Starting Quantization-Aware Training ---")
    qmodel.fit(X_train, y_train,
               batch_size=32,
               epochs=40,
               validation_data=(X_test, y_test),
               callbacks=[checkpoint, reduce_lr])

    print("\n--- Final Evaluation (Best Model) ---")
    best_qmodel = keras.models.load_model(MODEL_QAT_PATH, 
                                          custom_objects={'QConv2D': QConv2D, 'QActivation': QActivation, 'QDense': QDense})
    loss, acc = best_qmodel.evaluate(X_test, y_test)
    print(f"Final Quantized Accuracy: {acc*100:.2f}%")