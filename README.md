🧠 Brain Tumor Detection System

A deep learning-powered web application that detects brain tumors from MRI images using Convolutional Neural Networks (CNN) and Transfer Learning. The system provides real-time predictions through an interactive Streamlit interface.

🚀 Live Demo

🔗 Add your deployed app link here (Streamlit Cloud)

Example:
https://your-app-name.streamlit.app

📌 Project Overview

Brain tumor detection is a critical task in medical imaging. This project leverages deep learning to assist in early detection by classifying MRI scans into:

- ✅ Tumor Detected  
- ❌ No Tumor  

The system aims to support medical professionals by providing fast, AI-assisted predictions.

🧠 Model Architecture

This project uses a **Convolutional Neural Network (CNN)** with optional **Transfer Learning (MobileNetV2)**.

🔹 Key Features:
- Pretrained model: MobileNetV2 (ImageNet weights)
- Input shape: 224 × 224 × 3
- Output: Binary classification (Sigmoid activation)
- Loss Function: Binary Crossentropy
- Optimizer: Adam
- Regularization: Dropout to prevent overfitting

📊 Model Performance

| Metric       | Value       |
|--------------|-------------|
| Accuracy     | ~95%+       |
| Loss         | Low         |
| Type         | Binary Classification |

> Note: Accuracy may vary depending on dataset split and training conditions.

📂 Dataset

Dataset sourced from Kaggle:

🔗 https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection

Dataset Structure:

brain_tumor_dataset/
│
├── yes/ → Tumor images
├── no/ → Non-tumor images


⚙️ Tech Stack

- **Frontend:** Streamlit  
- **Backend / Model:** TensorFlow, Keras  
- **Image Processing:** OpenCV, PIL  
- **Language:** Python  

⚙️ Installation & Setup

1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/brain-tumor-detection.git
cd brain-tumor-detection

python -m venv venv

source venv/bin/activate     # Mac/Linux
venv\Scripts\activate        # Windows

pip install -r requirements.txt

streamlit run app.py
