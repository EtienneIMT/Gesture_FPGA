# (Dans src/train.py)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

try:
    from brevitas.onnx import export_qonnx  # <-- LE BON NOM
except ImportError as e:
    print(f"🚨 ERREUR D'IMPORT : {e}")
    print("Échec de l'import de 'export_qonnx' depuis 'brevitas.onnx'.")
    exit(1)

import onnx

# Importe la NOUVELLE architecture Brevitas
try:
    from models.py_models.cnn_gesture_brevitas import GestureNet
except ModuleNotFoundError:
    print(
        "Erreur: Le fichier models/py_models/cnn_gesture_brevitas.py est introuvable."
    )
    exit(1)

# --- Configuration ---
DATA_DIR = "data/processed"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
# Nouveaux noms de sortie
MODEL_SAVE_PATH_PT = "models/cnn_gesture_brevitas_int8.pt"
MODEL_SAVE_PATH_ONNX = "models/cnn_gesture_brevitas_int8.onnx"  # C'est le fichier QONNX

IMG_SIZE, N_CHANNELS = 64, 1
BATCH_SIZE, EPOCHS, LR = 64, 2, 1e-3  # Entraîne un peu plus longtemps
ONNX_OPSET = 17

# --- 1. Préparation du Dataset ---
print("--- Préparation du Dataset ---")
# Utilise les transformations avec augmentation
train_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Grayscale(num_output_channels=N_CHANNELS),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)
val_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Grayscale(num_output_channels=N_CHANNELS),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)
# ... (Code de chargement des données train_loader, val_loader reste identique) ...
try:
    train_data = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_data = datasets.ImageFolder(VAL_DIR, transform=val_transform)
    num_classes = len(train_data.classes)
    print(f"Nombre de classes: {num_classes}")
    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )
except Exception as e:
    print(f"Erreur chargement données: {e}")
    exit()

# --- 2. Initialisation du Modèle (Brevitas) ---
print("\n--- Initialisation du Modèle Brevitas (QAT) ---")
model = GestureNet(num_classes=num_classes)  # Instancie le modèle Brevitas
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Utilisation du device: {device}")

# --- 3. Boucle d'Entraînement (QAT) ---
print("\n--- Début de l'Entraînement QAT ---")
# ... (La boucle d'entraînement est IDENTIQUE à ton script 'train_model.py') ...
best_val_accuracy = 0.0
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for i, (imgs, labels) in enumerate(train_loader):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(
        f"[QAT] Epoch {epoch+1}/{EPOCHS} terminé. Loss moyen: {running_loss / len(train_loader):.4f}"
    )

    # Validation
    model.eval()
    correct_val, total_val = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
    val_accuracy = 100 * correct_val / total_val
    print(f"[QAT] Epoch {epoch+1} Validation Accuracy: {val_accuracy:.2f}%")

    if val_accuracy > best_val_accuracy:
        print("  -> Nouvelle meilleure précision ! Sauvegarde...")
        best_val_accuracy = val_accuracy
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH_PT), exist_ok=True)
        torch.save(model.state_dict(), MODEL_SAVE_PATH_PT)

print("--- Entraînement QAT Terminé ---")

# --- 4. Export vers QONNX (la clé !) ---
print(f"\n--- Export du Meilleur Modèle vers QONNX ---")
try:
    # Charge le meilleur modèle sauvegardé
    model = GestureNet(num_classes=num_classes)  # Recrée l'architecture
    model.load_state_dict(torch.load(MODEL_SAVE_PATH_PT))
    model.eval()

    dummy_input = torch.randn(1, N_CHANNELS, IMG_SIZE, IMG_SIZE)

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH_ONNX), exist_ok=True)

    # Utilise l'exportateur Brevitas
    export_qonnx(  # <-- LE BON NOM
        model,
        input_t=dummy_input,
        export_path=MODEL_SAVE_PATH_ONNX,
        opset_version=ONNX_OPSET,
    )

    print(f"Modèle QONNX (INT8) exporté dans: {MODEL_SAVE_PATH_ONNX}")

    # Vérification ONNX
    import onnx

    onnx_model = onnx.load(MODEL_SAVE_PATH_ONNX)
    onnx.checker.check_model(onnx_model)
    print("✅ Vérification QONNX réussie.")

except Exception as e:
    print(f"🚨 Erreur lors de l'export QONNX: {e}")
