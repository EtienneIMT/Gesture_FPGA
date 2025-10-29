import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as cfg
from finn.builder.build_dataflow_config import DataflowOutputType
import os

# --- Configuration ---
QONNX_MODEL_IN = "models/cnn_gesture_brevitas_int8.onnx" # Ton QONNX
output_dir = "finn_build/brevitas_int8_uz"
zynq_part = "xczu7ev-ffvc1156-2-e" # Ton Part Number
board = "ultrazed_eg_iocc"          
target_clk_period = 10.0            # 10.0 ns = 100 MHz

# --- Définition du Build (API Corrigée) ---
generate_outputs_list = [
    DataflowOutputType.ESTIMATE_REPORTS,
    DataflowOutputType.STITCHED_IP,
    DataflowOutputType.RTLSIM_PERFORMANCE,
    DataflowOutputType.BITFILE,
    DataflowOutputType.PYNQ_DRIVER,
    DataflowOutputType.DEPLOYMENT_PACKAGE,
]

# --- CORRECTION: Constructeur DataflowBuildConfig ---
# Respecte l'ordre positionnel que les erreurs nous ont indiqué
build_config = cfg.DataflowBuildConfig(
    
    # --- Arguments Positionnels (dans le bon ordre) ---
    output_dir,                 # 1er arg: output_dir
    QONNX_MODEL_IN,             # 2ème arg: model_filename
    target_clk_period,          # 3ème arg: synth_clk_period_ns
    generate_outputs_list,      # 4ème arg: generate_outputs
    
    # --- Arguments par mot-clé (le reste) ---
    fpga_part           = zynq_part,
    board               = board,
    auto_fifo_depths    = True
)

# --- Lancement du Build ---
print(f"--- Démarrage du Build FINN pour {QONNX_MODEL_IN} ---")
print(f"    Cible : {zynq_part} @ {target_clk_period} ns")
print(f"    Sortie : {output_dir}")
print(f"    Sorties demandées : {[o.name for o in generate_outputs_list]}")

try:
    # --- CORRECTION: Utiliser la bonne fonction ET les bons arguments ---
    # build_dataflow_cfg prend 2 arguments: le chemin du modèle ET l'objet config
    build.build_dataflow_cfg(QONNX_MODEL_IN, build_config) 
    
except Exception as e:
    print(f"\n🚨 LE BUILD A ÉCHOUÉ : {e}")
    print("\n--- DÉBOGAGE ---")
    print("Vérifie les logs dans les sous-dossiers de /finn_build/...")
    print("Cause la plus probable : Problème de licence Vivado/Vitis ou ressources FPGA.")
    exit(1) # Quitte avec une erreur si le build échoue

print("\n--- Build FINN Terminé ---")
print(f"✅ Résultats disponibles dans : {output_dir}")
print("Les fichiers de déploiement (bitstream, driver) sont dans le sous-dossier 'deploy'.")