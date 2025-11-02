# 3_convert_hls.py
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
def create_hls_config():
    config = {}
    config['ProjectName'] = 'hls_gesture_model'
    config['OutputDir'] = HLS_PROJECT_PATH
    config['Part'] = 'xczu3eg-sbva484-1-e' # Part Cible (UltraZed-EG)
    config['ClockPeriod'] = 10 # ns (Cible 100MHz)
    config['IOType'] = 'io_stream' # IMPORTANT: pour AXI-Stream et DMA
    
    # Stratégie de précision
    # Nous utilisons 'ap_fixed<8,3>' : 8 bits au total, 3 bits pour la partie entière
    # CELA DOIT ÊTRE AJUSTÉ en fonction de vos quantificateurs QKeras et de l'analyse 'profile'
    config['Model'] = {
        'Precision': 'ap_fixed<8,3>', 
        'ReuseFactor': 1, # Facteur de réutilisation = 1 -> Full parallélisme (max performance, max ressources)
        'Strategy': 'Latency' # Optimiser pour la latence
    }

    # Vous pouvez affiner la précision et le ReuseFactor pour chaque couche ici
    # Exemple :
    # config['LayerName'] = {
    #     'conv1': {'Precision': 'ap_fixed<8,1>', 'ReuseFactor': 4},
    #     'fc1': {'Precision': 'ap_fixed<10,4>', 'ReuseFactor': 1}
    # }
    
    return config

# --- Main script ---
if __name__ == "__main__":
    # 1. Re-créer les objets QKeras pour le chargement
    custom_objects = {}
    for layer_type in [QConv2D, QDense, QActivation, quantized_bits, quantized_relu]:
        custom_objects[layer_type.__name__] = layer_type

    # 2. Charger le modèle QAT
    print(f"Chargement du modèle quantisé depuis {MODEL_QAT_PATH}...")
    model = keras.models.load_model(MODEL_QAT_PATH, custom_objects=custom_objects)
    model.summary()
    
    # 3. Créer la configuration HLS
    config = create_hls_config()
    print("\nConfiguration HLS utilisée :")
    print(yaml.dump(config, default_flow_style=False))

    # 4. Convertir le modèle
    print("\nLancement de la conversion HLS4ML...")
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=config['OutputDir'],
        part=config['Part'],
        clock_period=config['ClockPeriod'],
        io_type=config['IOType']
    )
    
    print("Conversion terminée.")
    
    # 5. Profiler le modèle (estimation des ressources et types)
    # C'est un script intermédiaire crucial !
    print("\n--- Profilage du modèle (estimation) ---")
    profile = types_hlsmodel(hls_model)
    # Affichez le profil pour voir si vos types 'ap_fixed<...>' débordent
    print(profile)
    # hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file='model_hls.png')

    # 6. Sauvegarder le modèle hls4ml compilé (pour le script de build)
    hls_model.write()
    print(f"Projet HLS généré dans {HLS_PROJECT_PATH}")

# Lancement de la synthèse Vitis HLS
hls_model.build(
    csim=True, 
    synth=True, 
    export=True, 
    vsynth=True # Utiliser Vitis HLS
)