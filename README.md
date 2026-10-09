# Satellite-Damage-Classification-CNN
CNN based satellite imagery classification for identifying destroyed building, destroyed vehicles, and military vehicles. 
# Satellite Imagery Classification for Damage & Target Assessment using Custom CNN

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end Computer Vision pipeline built with TensorFlow/Keras to classify high-resolution satellite imagery tiles. The project focuses on detecting conflict-related structural damage, military equipment, and terrain features.

---

## 📌 Project Overview
* Dataset: 4,500 satellite tiles (100x100 pixels, RGB).
* Classes (5 Categories): Aircraft, Damaged Buildings, Damaged Vehicles, Military Assets, and No Target/Background.
* Architecture: Custom 5-class Convolutional Neural Network (CNN) without Batch Normalization.
* Key Achievements:
  * Test Accuracy: 86.0%
  * Validation Accuracy: 88.5%
  * Training Accuracy: 94.2%
  * Average Inference Speed: ~12-18 ms per image (~60 FPS).

---

## ⚙️ Key Technical Features
* Data Preprocessing: Bilinear interpolation resizing to 100x100 resolution, pixel rescaling (1./255), and stratified 70/15/15 data splitting.
* Data Augmentation: Integrated dynamic Keras layers (RandomFlip, RandomRotation, RandomZoom) to prevent overfitting.
* Optimization: Adam Optimizer ($\text{LR} = 0.001$) paired with Sparse Categorical Cross-Entropy Loss.
* Classification Logic: Multi-class probability vectors converted via Argmax.

---

## 📁 Repository Structure
`text
├── data/                  # Sample satellite tiles/directories
├── models/                # Saved trained model weights (.h5 / .keras)
├── notebooks/             # Data exploration & training scripts (.ipynb)
├── app.py                 # Interactive Web Dashboard (Streamlit / Folium)
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation

## 🚀 How to Run locally

1. Clone the repository:
   ```bash
   git clone https://github.com/majdyassar435-oss/Satellite-Damage-Classification-CNN.git
   cd Satellite-Damage-Classification-CNN
   ```
2. Install debendencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the interactive APP:
   ```bash
   streamlit run app.py
   ```
    
 ## 📜 License
 
​Distributed under the MIT License. See `LICENSE` for more information.
   
