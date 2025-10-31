import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as cfg
from finn.builder.build_dataflow_config import DataflowOutputType
import os

# --- Importations pour les étapes manuelles ---
from qonnx.core.modelwrapper import ModelWrapper
from finn.builder.build_dataflow_steps import (
    step_qonnx_to_finn,
    step_tidy_up,
    step_streamline,
    step_convert_to_hw,
    # step_create_dataflow_partition, # On n'utilise PAS celle-ci (buggée)
    step_target_fps_parallelization,
    step_apply_folding_config,
    step_generate_estimate_reports,
    step_hw_codegen,
    step_hw_ipgen,
    step_set_fifo_depths,
)
from qonnx.transformation.general import GiveUniqueNodeNames

# --- IMPORT POUR LE PARTITIONNEMENT MANUEL (LE BON) ---
from finn.transformation.fpgadataflow.create_dataflow_partition import CreateDataflowPartition

# --- Configuration ---
# Assure-toi que ce fichier est le QONNX de ton CNN (celui qui plante)
QONNX_MODEL_IN = "models/cnn_gesture_brevitas_int8.onnx"
output_dir = "finn_build/simple_test" 
zynq_part = "xczu7ev-ffvc1156-2-e" 
board = "ultrazed_eg_iocc"          
target_clk_period = 10.0            

# --- Définition du Build (ancienne API) ---
generate_outputs_list = [ DataflowOutputType.ESTIMATE_REPORTS ]

build_config = cfg.DataflowBuildConfig(
    output_dir, 
    QONNX_MODEL_IN, 
    target_clk_period,
    generate_outputs_list,
    fpga_part           = zynq_part,
    board               = board,
    auto_fifo_depths    = True
)

# --- Lancement du Build Manuel (Simple) ---
print(f"--- Démarrage du Build FINN SIMPLE pour {QONNX_MODEL_IN} ---")
print(f"    Sortie : {output_dir}")

try:
    model = ModelWrapper(QONNX_MODEL_IN) 
    
    print("Étape 1/8 : step_qonnx_to_finn...")
    model = step_qonnx_to_finn(model, build_config)
    print("Étape 2/8 : step_tidy_up...")
    model = step_tidy_up(model, build_config)
    print("Étape 3/8 : step_streamline...")
    model = step_streamline(model, build_config)
    print("Étape 4/8 : step_convert_to_hw...")
    model = step_convert_to_hw(model, build_config)
    print("Étape 5/8 : Application du correctif GiveUniqueNodeNames...")
    model = model.transform(GiveUniqueNodeNames())
    
    # --- CORRECTION : PARTITIONNEMENT MANUEL ---
    print("Étape 6/8 : Partitionnement manuel (CreateDataflowPartition)...")
    # Cette étape remplace 'step_create_dataflow_partition'
    model = model.transform(CreateDataflowPartition())
    print("Partitionnement manuel réussi.")
    
    print("Étape 7/8 : Définition du parallélisme...")
    model = step_target_fps_parallelization(model, build_config)
    model = step_apply_folding_config(model, build_config)

    print("Étape 8/8 : Génération des IP HLS...")
    model = step_hw_codegen(model, build_config) 
    model = step_hw_ipgen(model, build_config) 
    model = step_generate_estimate_reports(model, build_config) 

except Exception as e:
    print(f"\n🚨 LE BUILD A ÉCHOUÉ : {e}")
    print("\n--- DÉBOGAGE ---")
    print("Vérifie les logs dans les sous-dossiers de /finn_build/simple_test...")
    exit(1)

print("\n--- Build FINN Simple Terminé ---")
print(f"✅ Succès ! Les IPs HLS ont été générées dans : {output_dir}")
print("Tu peux maintenant lancer le script de build complet (build_finn_accelerator.py).")