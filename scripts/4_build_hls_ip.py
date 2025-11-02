# 4_build_hls_ip.py
import hls4ml
import os
from settings import HLS_PROJECT_PATH

print(f"Chargement du projet HLS depuis {HLS_PROJECT_PATH}...")

# 1. Charger le modèle hls4ml précédemment converti
# Assurez-vous que le chemin pointe vers le fichier .json de configuration généré par le script 3
config_file = os.path.join(HLS_PROJECT_PATH, 'hls4ml_config.yml') 
# Non, hls_model.save() crée un tarball ou un dossier.
# Nous devons re-charger le hls_model à partir du code source.
# Le plus simple est de ré-exécuter la conversion...
# MAIS hls4ml 0.8.1 permet de charger un projet !

# Correction: hls4ml.model.HLSModel.load_model() n'existe pas dans 0.8.1.
# Le flux standard est d'appeler .build() directement après .convert() dans le script 3.
# Si vous voulez les séparer, vous devez re-créer l'objet hls_model.

# -----
# NOTE : Idéalement, combinez ce script avec la fin du `3_convert_hls.py`
# Si vous devez les séparer, vous devez re-faire la conversion :
# -----
print("Ce script doit être lancé dans un environnement VIVADO/VITIS sourcé.")
print("Il est recommandé d'ajouter l'étape 'build' à la fin du script 3.")
print("Tentative de re-conversion et build...")

# Solution (combine 3 et 4) :
# ... (Copier le contenu du script 3 jusqu'à `hls_model = hls4ml.converters.convert_from_keras_model(...)` ) ...
# hls_model.convert() # Assurez-vous qu'il a été converti
# ...

# ON SUPPOSE QUE LE SCRIPT 3 A DÉJÀ ÉTÉ LANCÉ
# On charge l'objet hls_model depuis le script précédent (s'il est exécuté en série)
# SI EXÉCUTÉ SÉPARÉMENT : rechargez le modèle Keras et reconvertissez.
print("Rechargement du modèle HLS (nécessite une re-conversion en mémoire)...")
# (Code du script 3 pour charger keras et recréer hls_model)
# ... (ici, on suppose que hls_model est en mémoire)
# hls_model = ... (résultat du script 3)

# LA VRAIE FAÇON DE FAIRE (en supposant que vous exécutez après le script 3):
# hls_model = ... (résultat de la conversion)

# --- Début du script 4 (en supposant que hls_model est chargé) ---
# import hls_model # Ne fonctionne pas comme ça

print("ERREUR: Le 'build' doit être appelé sur l'objet 'hls_model' créé par le script 3.")
print("Veuillez ajouter les lignes suivantes à la fin de '3_convert_hls.py' :")
print("""
# --- Ajout pour Build (anciennement Script 4) ---
print("\\n--- Lancement de la synthèse HLS (Build) ---")
print("Cela peut prendre plusieurs minutes...")

# csim = simulation C (rapide)
# synth = synthèse HLS (long)
# cosim = co-simulation C/RTL (vérification, long)
# export = empaquetage de l'IP pour Vivado (essentiel)
report = hls_model.build(
    csim=True, 
    synth=True, 
    cosim=False, # Désactivé pour accélérer, activez pour une vérification complète
    export=True, 
    vsynth=True # Utiliser Vitis HLS
)

print("Build HLS terminé.")
print("Rapport de synthèse :")
print(report)

print("\\n--- L'IP est prêt ! ---")
print("Vous pouvez trouver l'IP exporté dans le dossier :")
print(f"{HLS_PROJECT_PATH}/{config['ProjectName']}_prj/solution1/impl/ip")
""")