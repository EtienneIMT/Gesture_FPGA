"""
Train the baseline floating-point (float32) CNN model for gesture recognition.
"""

import sys
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

# Ensure the root directory is on the path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.settings import INPUT_SHAPE, NUM_CLASSES, MODEL_FLOAT_PATH
from src.data_loader import load_data

def build_pro_model():
    """Builds a compact CNN architecture optimized for edge deployment."""
    model = Sequential(name="cnn_float_pro")
    model.add(Input(shape=INPUT_SHAPE))
    
    # Block 1
    model.add(Conv2D(16, (3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(MaxPooling2D((2, 2))) 
    
    # Block 2
    model.add(Conv2D(32, (3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(MaxPooling2D((2, 2))) 
    
    # Block 3
    model.add(Conv2D(64, (3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    
    # Block 4
    model.add(Conv2D(64, (3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Flatten())
    
    model.add(Dense(64, use_bias=False))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    
    # Dropout for robust training
    model.add(Dropout(0.4))
    model.add(Dense(NUM_CLASSES, activation='softmax', name='output_softmax'))
    
    return model

if __name__ == "__main__":
    print("Loading data...")
    (X_train, y_train), (X_test, y_test) = load_data()
    
    # Data Augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False 
    )
    datagen.fit(X_train)

    model = build_pro_model()
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Callbacks for learning rate scheduling and saving the best model
    checkpoint = ModelCheckpoint(MODEL_FLOAT_PATH, 
                                 monitor='val_accuracy', 
                                 verbose=1, 
                                 save_best_only=True, 
                                 mode='max')
                                 
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=5, min_lr=0.00001, verbose=1)

    print("\n--- Starting Baseline Training ---")
    model.fit(datagen.flow(X_train, y_train, batch_size=32),
              epochs=50,
              validation_data=(X_test, y_test),
              callbacks=[checkpoint, reduce_lr])

    print("\nLoading the best saved model for final evaluation...")
    best_model = keras.models.load_model(MODEL_FLOAT_PATH)
    loss, acc = best_model.evaluate(X_test, y_test)
    print(f"Final Validation Accuracy: {acc*100:.2f}%")