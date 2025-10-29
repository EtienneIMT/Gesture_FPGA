# scripts/train_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# Importe la définition du modèle depuis le fichier models/cnn_model.py
from models.py_models.cnn_gesture_v1 import GestureNet 

# --- Configuration ---
DATA_DIR = "data/processed"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
MODEL_SAVE_PATH_PT = "models/cnn_gesture_v1.pt" # Pour les poids PyTorch
MODEL_SAVE_PATH_ONNX = "models/cnn_gesture_v1.onnx" # Pour le modèle ONNX

IMG_SIZE = 64 # Doit correspondre à la Resize et au calcul Linear
N_CHANNELS = 1 # Car on utilise Grayscale
BATCH_SIZE = 64
EPOCHS = 6 # Augmente si nécessaire
LEARNING_RATE = 1e-3
ONNX_OPSET = 18 # Utilise une version récente compatible FINN

# --- 1. Préparation du Dataset ---
print("--- Préparation du Dataset ---")

# Define separate transforms for training (with augmentation) and validation (without)
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3), # add if you temporarily convert to RGB then back
    transforms.Grayscale(num_output_channels=N_CHANNELS),
    transforms.RandomRotation(15),  # Rotate +/- 15 degrees
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)), # Shift and zoom slightly
    transforms.RandomPerspective(distortion_scale=0.2, p=0.4), # Add slight perspective shifts
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) 
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=N_CHANNELS), # Assure 1 canal
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) # Pour N_CHANNELS=1
])

try:
    train_data = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_data = datasets.ImageFolder(VAL_DIR, transform=val_transform)

    # Détermine le nombre de classes dynamiquement
    num_classes = len(train_data.classes)
    print(f"Nombre de classes détectées: {num_classes}")
    print(f"Classes: {train_data.classes}") # Affiche les labels trouvés

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
except FileNotFoundError as e:
    print(f"Erreur: Répertoire de données non trouvé : {e}")
    print("Assure-toi d'avoir exécuté le script de préparation du dataset.")
    exit()
except Exception as e:
    print(f"Erreur lors du chargement des données: {e}")
    exit()


# --- 2. Initialisation du Modèle, Perte, Optimiseur ---
print("\n--- Initialisation du Modèle ---")
model = GestureNet(num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Utilisation du device: {device}")

# --- 3. Boucle d'Entraînement ---
print("\n--- Début de l'Entraînement ---")
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
        if (i + 1) % 100 == 0: # Affiche le log toutes les 100 batches
            print(f'Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    print(f"Epoch {epoch+1} terminé. Loss moyen: {running_loss / len(train_loader):.4f}")

    # --- Évaluation à la fin de chaque époque (Optionnel mais recommandé) ---
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Validation Accuracy après Epoch {epoch+1}: {accuracy:.2f}%")

print("--- Entraînement Terminé ---")

# --- 4. Évaluation Finale (float32) ---
# L'évaluation est déjà faite à la dernière époque ci-dessus. 
# On affiche juste le résultat final.
print(f"\n--- Évaluation Finale sur Validation Set ---")
print(f"Accuracy (float32): {accuracy:.2f}%")

# --- 5. Sauvegarde et Export ---
print("\n--- Sauvegarde du Modèle ---")
try:
    # Sauvegarde des poids PyTorch
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH_PT), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH_PT)
    print(f"Poids PyTorch sauvegardés dans: {MODEL_SAVE_PATH_PT}")

    # Export vers ONNX
    model.eval() # Assure que le modèle est en mode évaluation pour l'export
    model.to("cpu") # Exporte sur CPU pour éviter les dépendances CUDA dans ONNX si possible

    # Crée une entrée factice (dummy input) avec la bonne taille
    dummy_input = torch.randn(1, N_CHANNELS, IMG_SIZE, IMG_SIZE) 

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH_ONNX), exist_ok=True)
    torch.onnx.export(
        model, 
        dummy_input, 
        MODEL_SAVE_PATH_ONNX, 
        export_params=True, # Inclut les poids entraînés
        opset_version=ONNX_OPSET, # Version compatible FINN/PyTorch récent
        do_constant_folding=True, # Optimisation
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes={'input' : {0 : 'batch_size'}, # Permet différentes tailles de batch
                      'output' : {0 : 'batch_size'}}
    )
    print(f"Modèle exporté vers ONNX dans: {MODEL_SAVE_PATH_ONNX}")

    # Vérification du modèle ONNX
    import onnx
    onnx_model = onnx.load(MODEL_SAVE_PATH_ONNX)
    onnx.checker.check_model(onnx_model)
    print("✅ Vérification ONNX réussie.")

except Exception as e:
    print(f"Erreur lors de la sauvegarde ou de l'export ONNX: {e}")