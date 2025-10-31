import onnx
import sys

# Charger le modèle final
final_model_path = "models/cnn_gesture_brevitas_int8_FINAL_FOR_HLS.onnx"
try:
    onnx_model = onnx.load(final_model_path)
except Exception as e:
    print(f"Erreur lors du chargement de {final_model_path}: {e}")
    sys.exit(1)

# Imprimer les entrées et sorties du graphe
print(f"--- Inspection de {final_model_path} ---")

print("🔬 Entrées du graphe :")
for inp in onnx_model.graph.input:
    print(f"  - {inp.name}")

print("\n🔬 Sorties du graphe :")
if not onnx_model.graph.output:
    print("  - ERREUR: Aucune sortie trouvée dans le graphe !")
for outp in onnx_model.graph.output:
    print(f"  - {outp.name}")

print("-----------------------------------------")