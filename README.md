# 🧠 Accélérateur FPGA pour la Reconnaissance de Gestes en Temps Réel

Ce projet démontre un **workflow complet de co-design matériel/logiciel** pour accélérer un réseau de neurones convolutifs (CNN) de reconnaissance de gestes de la main sur une plateforme embarquée Xilinx Zynq UltraScale+ MPSoC (Avnet UltraZed-EG).

L'objectif est de décharger l'inférence du CPU (ARM) vers le FPGA (PL) pour obtenir une **inférence à faible latence et économe en énergie**, adaptée aux applications embarquées (robotique, appareils intelligents, IHM).

---

## 🚀 Architecture et Flux de Travail

Ce projet ne se contente pas d'entraîner un modèle ; il le compile en matériel. Le flux de travail complet, de Keras à un bitstream FPGA, est le suivant :

1.  **Entraînement Keras (Float) :** Un CNN compact est d'abord entraîné avec Keras en `float32` pour établir une baseline de précision.
2.  **Entraînement QAT (QKeras) :** Le modèle est converti en **QKeras** et ré-entraîné (ou *fine-tuné*) en **Quantization-Aware Training (QAT)**. Cela adapte les poids du réseau à une arithmétique de faible précision (ex: `INT8`) que le FPGA peut calculer efficacement.
3.  **Export `.h5` :** Le modèle quantifié est sauvegardé au format `.h5`. **HLS4ML** est capable de lire ce fichier et d'interpréter directement les couches QKeras pour en déduire les types de données matériels (ex: `ap_fixed<8,2>`).
4.  **Synthèse HLS (HLS4ML) :** **HLS4ML** est utilisé pour convertir le graphe `.h5` en code **C++ HLS** optimisé. Il génère un projet complet prêt pour la synthèse.
5.  **Compilation Matérielle (Vitis HLS) :** **Vitis HLS** (appelé par le script de *build* HLS4ML) synthétise le C++ en un bloc **IP matériel** (RTL - Verilog/VHDL) prêt à être importé dans un design logique.
6.  **Intégration Système (Vivado) :** L'IP matériel est importé dans **Vivado** et intégré dans un *Block Design* Zynq MPSoC. Il est connecté au processeur (PS) via une interface **AXI-Lite** (pour le contrôle) et à la mémoire DDR via **AXI DMA** (pour le flux des pixels d'images). Un *bitstream* est alors généré.
7.  **Déploiement (PYNQ) :** L'application finale s'exécute sous **PYNQ** (Python sur Zynq) sur la carte **UltraZed-EG**. Le code Python (s'exécutant sur le CPU ARM) gère :
    * La capture vidéo via OpenCV.
    * Le pré-traitement de l'image (redimensionnement 64x64, normalisation).
    * L'envoi de l'image à l'accélérateur (PL) via le DMA.
    * La réception des résultats (logits) du PL.
    * Le post-traitement (Softmax en CPU) et l'affichage du geste reconnu.