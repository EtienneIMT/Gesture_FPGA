import onnx
import hls4ml

# === 1. Charger le modèle FINAL (déjà corrigé ET channels-last) ===
final_model_path = "models/cnn_gesture_brevitas_int8_FINAL_FOR_HLS.onnx"
print(f"🔍 Chargement du modèle ONNX final : {final_model_path}")
onnx_model = onnx.load(final_model_path)

# === 2. Génération de la configuration HLS4ML ===
print("⚙️ Génération de la configuration HLS4ML...")
config = hls4ml.utils.config_from_onnx_model(onnx_model, backend='Vivado')

# 'final_output_layer' est le nom que nous avons donné à l'étape 2
output_layer_name = "final_output_layer"
if output_layer_name in config['LayerName']:
    config['LayerName'][output_layer_name]['IOType'] = 'io_stream'
    print(f"✅ Configuration IOType=io_stream forcée pour '{output_layer_name}'.")
else:
    print(f"⚠️ ATTENTION: Impossible de trouver '{output_layer_name}' dans la config pour forcer l'IOType.")


# Optionnel : afficher la topologie détectée
print("🔍 Topologie du modèle détectée :")
print(config["LayerName"])

# === 3. Génération du projet HLS ===
output_dir = "hls4ml_prj"
print(f"🚀 Conversion vers HLS4ML (dossier : {output_dir})...")
hls_model = hls4ml.converters.convert_from_onnx_model(
    onnx_model,
    hls_config=config,
    output_dir=output_dir,
    backend='Vivado',
    io_type='io_stream',
    output_layers=["global_out"]
)

hls_model.compile()
print("✅ Conversion réussie ! Projet généré dans", output_dir)