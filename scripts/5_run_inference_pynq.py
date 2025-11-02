# 5_run_inference_pynq.py
# --- À EXÉCUTER SUR LA CARTE PYNQ (ULTRAZED) ---

import numpy as np
from pynq import Overlay, allocate
import cv2 # Pour la capture caméra
import time
from settings import * # Copiez settings.py sur la carte

# --- Fonctions utilitaires ---

def preprocess_image(frame):
    """ Prépare l'image de la caméra pour le DNN """
    # 1. Redimensionner
    img_resized = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
    
    # 2. Normaliser (si votre modèle a été entraîné sur [0,1])
    # img_normalized = img_resized / 255.0
    
    # IMPORTANT: HLS4ML attend des entiers (ap_fixed).
    # Si QKeras utilise quantized_bits(8, 0), il attend des entiers 8 bits.
    # Nous supposerons que le modèle HLS attend des UINT8 (0-255)
    
    # 3. Assurer le bon type
    # (Ajustez dtype selon la précision d'entrée de votre IP HLS)
    # ex: ap_fixed<8,3> -> pourrait nécessiter un float converti
    # ex: ap_uint<8> -> np.uint8
    img_processed = img_resized.astype(np.uint8) # Supposition : HLS prend UINT8
    
    return img_processed

def postprocess_output(output_buffer):
    """ Applique le Softmax sur la sortie (logits) du FPGA """
    # exp(x) / sum(exp(x))
    logits = output_buffer.astype(np.float32)
    e_x = np.exp(logits - np.max(logits)) # Soustraction pour stabilité numérique
    return e_x / e_x.sum(axis=0)

# --- Configuration PYNQ ---
OVERLAY_PATH = "gesture_recognition.bit"
GESTURE_CLASSES = ['Main ouverte', 'Poing', 'Pouce levé', 'Swipe Gauche', 'Swipe Droite'] # Vos 5 classes

print("Chargement de l'overlay PYNQ...")
overlay = Overlay(OVERLAY_PATH)

# Récupérer les drivers pour le DMA et l'IP
# Les noms (ex: 'axi_dma_0', 'hls_gesture_model_0') 
# doivent correspondre à votre Block Design Vivado
dma = overlay.axi_dma_0
hls_ip = overlay.hls_gesture_model_0 # Nom de l'IP HLS

# Allouer des tampons (buffers) en mémoire contiguë pour le DMA
# La taille doit correspondre EXACTEMENT à ce que l'IP attend
input_buffer = allocate(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
output_buffer = allocate(shape=(NUM_CLASSES,), dtype=np.int8) # Ou np.int16/32 selon la sortie HLS

print("Configuration PYNQ terminée. Lancement de la capture...")

# Ouvrir la caméra USB
cap = cv2.VideoCapture(0) # 0 est généralement la première caméra USB
if not cap.isOpened():
    raise IOError("Impossible d'ouvrir la caméra")

fps_list = []

try:
    while True:
        # 1. Capture Caméra (PS)
        ret, frame = cap.read()
        if not ret:
            break
            
        start_time_e2e = time.time() # Début du chronomètre End-to-End

        # 2. Prétraitement (PS)
        input_image = preprocess_image(frame)
        
        # 3. Copier l'image dans le buffer DMA
        np.copyto(input_buffer, input_image)

        # 4. Lancer l'inférence (DMA + PL)
        
        # Début du chronomètre Matériel
        # (Pour une mesure précise de la latence PL, utilisez des timers AXI)
        start_time_pl = time.time() 
        
        # Envoyer l'image au FPGA
        dma.sendchannel.transfer(input_buffer)
        
        # Récupérer la sortie du FPGA
        dma.recvchannel.transfer(output_buffer)
        
        # Démarrer l'IP HLS (s'il n'est pas auto-restart)
        # hls_ip.write(0x00, 1) # Démarre l'IP (adresse 0x00 = AP_START)
        
        # Attendre la fin des transferts
        dma.sendchannel.wait()
        dma.recvchannel.wait()
        
        latence_pl_dma = (time.time() - start_time_pl) * 1000 # en ms
        
        # 5. Post-traitement (PS)
        probabilities = postprocess_output(output_buffer)
        prediction = np.argmax(probabilities)
        confidence = probabilities[prediction]
        
        latence_e2e = (time.time() - start_time_e2e) * 1000 # en ms
        fps = 1000 / latence_e2e
        fps_list.append(fps)

        # 6. Affichage (Console UART)
        print(f"--- Geste Détecté ---")
        print(f">>> {GESTURE_CLASSES[prediction]} (Conf: {confidence:.2f})")
        print(f"Latence E2E: {latence_e2e:.2f} ms")
        print(f"Latence PL+DMA (approx): {latence_pl_dma:.2f} ms")
        print(f"FPS (inst): {fps:.1f}")
        print(f"FPS (avg): {np.mean(fps_list):.1f}\n")

        # (Optionnel) Afficher l'image sur un GUI
        # cv2.putText(frame, GESTURE_CLASSES[prediction], ...)
        # cv2.imshow('Gesture Recognition', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

except KeyboardInterrupt:
    print("\nArrêt du programme.")
finally:
    cap.release()
    cv2.destroyAllWindows()
    del input_buffer
    del output_buffer
    print("Ressources libérées.")