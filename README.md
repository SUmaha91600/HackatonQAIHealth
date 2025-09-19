# Classic Health Predictor & Segmentation Multi-Lesions on MRI

# Prototype éducatif — Streamlit + Scikit-learn + OpenCV + Scikit-image
⚠️ Non clinique — usage pédagogique uniquement

# 🇫🇷 Version Française
# 📌 Contexte & Objectifs

Ce projet a été développé dans le cadre d’un hackathon IA Santé.
Il illustre comment :

- Segmenter automatiquement des lésions visibles en IRM mammaire (multi-lésions, formes quelconques).

- Extraire des features géométriques & texturales à partir de chaque lésion.

- Prédire un score bénin/malin via des modèles classiques (Logistic Regression & SVM).

- Proposer une interface interactive avec Streamlit (sliders, upload d’images, heatmap, export CSV).

# 👉 Important : Le modèle de classification est entraîné sur le dataset Breast Cancer Wisconsin (histologie) et non sur des IRM. Les résultats n’ont aucune valeur clinique.

# 🏗️ Pipeline du projet
# 1.⁠ ⁠Données de référence

Dataset Breast Cancer Wisconsin.

- Features sélectionnées : mean radius, mean texture, mean perimeter, mean area, mean compactness.

- Modèles : Logistic Regression & SVM RBF, entraînés avec MinMaxScaler.

- Visualisation PCA 2D des patients + projection des lésions segmentées.

# 2.⁠ ⁠Segmentation IRM

Prétraitement (CLAHE, band-pass, lissage).

Seuils robustes (Percentile, Multi-Otsu).

Morphologie (ouverture, fermeture, remplissage).

Extraction de régions (aire, circularité, excentricité, contraste anneau…).

Score par lésion = aire × intensité × contraste × circularité.

# 3.⁠ ⁠Interface Streamlit

Onglet 1 : sliders pour ajuster les features.

Onglet 2 : upload IRM, segmentation multi-lésions, heatmap, export CSV.

Projection PCA 2D patients vs lésions.

# 🔬 Vision Quantique (futur)

QSVM / Quantum Kernel Estimation : classification dans des espaces de Hilbert riches.

QAOA / VQE : optimisation combinatoire des seuils/morpho.

Pipeline hybride : traitement d’image classique + backend quantique pour classification/optimisation.

# ⚙️ Installation & Lancement
git clone <repo_url>
cd <repo>
pip install -r requirements.txt
streamlit run Breast.py

# 📂 Structure
Breast.py           # Application principale Streamlit


# 🚧 Limites

Classifieur basé sur histologie (non IRM).

Pas de support DICOM.

Probabilités non calibrées.

Segmentation heuristique (pas d’UNet).

# ✅ Perspectives

Support DICOM + annotations médicales.

Calibration des probabilités.

Comparaison avec deep learning (UNet).

Implémentation QSVM/QAOA en simulation.

Études cliniques pilotes.

# 👥 Auteurs

Amelya, Sutharsan, Pierre Louis, Simon, David
Aivancity Hackathon — 18 septembre 2025

# 🇬🇧 English Version
# 📌 Context & Objectives

This project was developed during a Healthcare AI Hackathon.
It demonstrates how to:

Automatically segment visible breast MRI lesions (multi-lesions, arbitrary shapes).

Extract geometric & textural features for each lesion.

Predict benign/malignant score using classic models (Logistic Regression & SVM).

Provide an interactive Streamlit interface (sliders, upload, heatmap, CSV export).

👉 Important: The classifier is trained on the Breast Cancer Wisconsin dataset (histology), not MRI. Results have no clinical value.

# 🏗️ Project Pipeline
# 1.⁠ ⁠Reference Data

Breast Cancer Wisconsin dataset.

Selected features: mean radius, mean texture, mean perimeter, mean area, mean compactness.

Models: Logistic Regression & SVM RBF, trained with MinMaxScaler.

PCA 2D visualization of patients + projection of segmented lesions.

# 2.⁠ ⁠MRI Segmentation

Preprocessing (CLAHE, band-pass filter, smoothing).

Robust thresholds (Percentile, Multi-Otsu).

Morphology (opening, closing, hole filling).

Region extraction (area, circularity, eccentricity, ring contrast…).

Lesion score = area × intensity × contrast × circularity.

# 3.⁠ ⁠Streamlit Interface

Tab 1: sliders to adjust features.

Tab 2: upload MRI, multi-lesion segmentation, heatmap, CSV export.

PCA 2D projection (patients vs lesions).

# 🔬 Quantum Vision (future work)

QSVM / Quantum Kernel Estimation: classification in rich Hilbert spaces.

QAOA / VQE: combinatorial optimization of thresholds/morphology.

Hybrid pipeline: classical image processing + quantum backend for classification/optimization.

# ⚙️ Installation & Run
git clone <repo_url>
cd <repo>
pip install -r requirements.txt
streamlit run Breast.py

# 📂 Structure
Breast.py           # Main Streamlit app


# 🚧 Limitations

Classifier trained on histology (not MRI).

No DICOM support.

Probabilities not calibrated.

Segmentation based on heuristics (no UNet).

# ✅ Next Steps

Add DICOM support + medical annotations.

Probability calibration.

Compare with deep learning (UNet).

Implement QSVM/QAOA in simulation.

Pilot clinical studies.

# 👥 Authors

Amelya, Sutharsan, Pierre Louis, Simon, David
Aivancity Hackathon — September 18, 2025
