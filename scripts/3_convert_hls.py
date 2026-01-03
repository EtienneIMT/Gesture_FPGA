import hls4ml
from hls4ml.model.profiling import types_hlsmodel
import tensorflow as tf
from tensorflow import keras
import qkeras
import numpy as np
import yaml
from qkeras import QConv2D, QDense, QActivation
from qkeras.quantizers import quantized_bits, quantized_relu

from settings import *

# --- Configuration HLS4ML ---
def create_hls_config(model):
    
    # 1. Générer la structure de base
    config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    
    print("Structure de base générée. Application des optimisations...")

    # 2. Paramètres Globaux
    config["ProjectName"] = "hls_gesture_model"
    config["OutputDir"] = HLS_PROJECT_PATH
    config["Part"] = "xczu3eg-sbva484-1-e"
    config["ClockPeriod"] = 10
    config["IOType"] = "io_stream"

    # 3. Stratégie Globale
    config["Model"] = {
        "Precision": "ap_fixed<10,4>",
        "ReuseFactor": 512, # On force le mode série
        "Strategy": "Resource"
    }

    # 4. Forçage couche par couche (Correction ici !)
    for layer in config['LayerName'].keys():
        config['LayerName'][layer]['ReuseFactor'] = 512
        config['LayerName'][layer]['Strategy'] = 'Resource'
        
        # Optimisation spécifique pour les convolutions
        if 'conv' in layer:
            config['LayerName'][layer]['ReuseFactor'] = 512
            # CORRECTION : On utilise 'LineBuffer' qui est le standard pour le streaming
            config['LayerName'][layer]['ConvImplementation'] = 'LineBuffer' 

    # Vérification visuelle
    if 'fc1' in config['LayerName']:
        print(f"Configuration pour 'fc1' : {config['LayerName']['fc1']}")

    return config


# --- Main script ---
if __name__ == "__main__":
    # 1. Re-créer les objets QKeras
    custom_objects = {}
    for layer_type in [QConv2D, QDense, QActivation, quantized_bits, quantized_relu]:
        custom_objects[layer_type.__name__] = layer_type

    # 2. Charger le modèle
    print(f"Chargement du modèle quantisé depuis {MODEL_QAT_PATH}...")
    model = keras.models.load_model(MODEL_QAT_PATH, custom_objects=custom_objects)
    model.summary()

    # 3. Créer la configuration
    config = create_hls_config(model)
    
    print("\nConfiguration HLS utilisée (Global) :")
    print(config["Model"])

    # 4. Convertir
    print("\nLancement de la conversion HLS4ML...")
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=config["OutputDir"],
        part=config["Part"],
        clock_period=config["ClockPeriod"],
        io_type=config["IOType"],
    )

    print("Conversion terminée.")

    # 5. Compiler
    hls_model.write()
    print(f"Projet HLS généré dans {HLS_PROJECT_PATH}")

    # Lancement de la synthèse Vitis HLS
    print("Lancement de la synthèse Vitis HLS (Cela va prendre quelques minutes)...")
    hls_model.build(
        csim=False, 
        synth=True, 
        export=True, 
        vsynth=True 
    )
    
    print("Synthèse terminée. Vérifiez le tableau 'Synthesis Report' ci-dessus !")