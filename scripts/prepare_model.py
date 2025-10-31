import onnx
from onnx import helper, TensorProto

# Importations QONNX
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.channels_last import ConvertToChannelsLastAndClean
from qonnx.util.cleanup import cleanup_model  # <- L'outil de nettoyage officiel
from qonnx.transformation.gemm_to_matmul import GemmToMatMul

print("--- ÉTAPE A : CHARGEMENT ET NETTOYAGE (via QONNX) ---")

# === 1. Charger le modèle ORIGINAL ===
onnx_model_path = "models/cnn_gesture_brevitas_int8.onnx"
print(f"🔍 Chargement du modèle ONNX : {onnx_model_path}")
onnx_model = onnx.load(onnx_model_path)


# === 2. Forçage de la forme d'entrée (NCHW) ===
# Le nettoyage a besoin de connaître la forme d'entrée
input_name = "inp.1"
input_shape = [1, 1, 64, 64] # Forme NCHW originale

input_tensor = next((x for x in onnx_model.graph.input if x.name == input_name), None)
if input_tensor is None:
    print(f"⚠️ Entrée {input_name} non trouvée, ajout manuel.")
    input_tensor = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, input_shape)
    onnx_model.graph.input.append(input_tensor)
else:
    print(f"✅ Entrée {input_name} trouvée. Forçage de la forme à {input_shape}.")
    input_tensor.type.tensor_type.ClearField('shape')
    input_tensor.type.tensor_type.elem_type = TensorProto.FLOAT
    for dim_val in input_shape:
         input_tensor.type.tensor_type.shape.dim.extend([
             onnx.TensorShapeProto.Dimension(dim_value=dim_val)
         ])

# === 3. Nettoyage et Inférence de Forme QONNX ===
print("🧹 Nettoyage du modèle avec qonnx.util.cleanup.cleanup_model...")
# Emballer le modèle
model_to_clean = ModelWrapper(onnx_model)

# Appliquer le nettoyage. 
# Ceci va (correctement) :
#  - Retirer les initializers des entrées
#  - Exécuter une inférence de forme robuste
#  - Propager les formes à travers tout le graphe
cleaned_model = cleanup_model(model_to_clean)
print("✅ Modèle NCHW nettoyé et formes inférées.")


print("🔄 Conversion de Gemm en MatMul...")
cleaned_model = cleaned_model.transform(GemmToMatMul())
print("✅ Transformation GemmToMatMul réussie.")


print("\n--- ÉTAPE B : CONVERSION VERS CHANNELS-LAST (NHWC) ---")

# === 4. Conversion en "channels-last" ===
try:
    # Nous avons déjà un ModelWrapper "cleaned_model"
    print("🔄 Application de la transformation 'ConvertToChannelsLastAndClean'...")
    model_channels_last = cleaned_model.transform(
        ConvertToChannelsLastAndClean(make_input_channels_last=True)
    )
    
    print("✅ Transformation réussie.")

    print("✏️  Vérification finale des noms de nœuds (post-transformation)...")
    unnamed_count = 0
    final_graph = model_channels_last.model.graph
    output_tensor_name = "global_out"
    output_node_found = False

    for i, node in enumerate(final_graph.node):
        # 1. Renommer les nœuds vides
        if not node.name: 
            node.name = f"{node.op_type}_{i}_unnamed_post_FIXED"
            unnamed_count += 1

        # 2. Trouver et nommer la couche de sortie finale
        if output_tensor_name in node.output:
            print(f"✅ Nœud de sortie (type {node.op_type}) trouvé. Renommage en 'final_output_layer'.")
            node.name = "final_output_layer"
            output_node_found = True

    if not output_node_found:
        print(f"⚠️ ATTENTION: Impossible de trouver le nœud qui produit '{output_tensor_name}'!")

    print(f"✅ {unnamed_count} nœuds sans nom ont été renommés.")
    
    # === 5. Sauvegarde du modèle final ===
    final_model_path = "models/cnn_gesture_brevitas_int8_FINAL_FOR_HLS.onnx"
    model_channels_last.save(final_model_path)
    print(f"🎉 Modèle final sauvegardé : {final_model_path}")

except Exception as e:
    print(f"❌ Échec de la transformation QONNX : {e}")
    print("Sauvegarde du modèle 'nettoyé' pour débogage...")
    cleaned_model.save("models/cnn_gesture_brevitas_int8_CLEANED_DEBUG.onnx")
    raise