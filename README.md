
-----

# 🌱 Plant Disease Detection System

An AI-powered system for detecting plant diseases from leaf images using Convolutional Neural Networks (CNN).

**Project Link:-** [https://plantdiseasedetectionsystems.streamlit.app/](https://plantdiseasedetectionsystems.streamlit.app/)

<img width="1831" height="928" alt="image" src="https://github.com/user-attachments/assets/9ee9be58-8c01-4c9b-bbcf-97689f095bf4" />


## 📋 Project Overview

This project uses deep learning to classify plant diseases from leaf images. The system is trained on the PlantVillage dataset and can identify 5 different classes:

  - **Apple Apple Scab**
  - **Apple Black Rot**
  - **Apple Cedar Apple Rust**
  - **Apple Healthy**
  - **Tomato Bacterial Spot**

## 🚀 Features

  - **Advanced Deep Learning Models**: Utilizes state-of-the-art architectures like **EfficientNetV2** and **ResNet50V2**.
  - **High-Accuracy via Fine-Tuning**: Implements a two-stage fine-tuning strategy to adapt powerful pre-trained models for maximum accuracy on the plant leaf dataset.
  - **Optimized Data Pipeline**: Employs the `tf.data` API with `.cache()` and `.prefetch()` to eliminate CPU bottlenecks and ensure high-speed, GPU-accelerated training.
  - **Ensemble-Ready System**: Designed to train and save multiple models, allowing for a final high-performance ensemble model.
  - **GUI Application**: User-friendly desktop interface built with Tkinter.
  - **Web Deployment**: Interactive web application built with Streamlit.
  - **Disease Information**: Provides symptoms and treatment recommendations for predicted diseases.

## 📁 Project Structure

```
PROJECT/
├── main.py                     # GUI application (tkinter)
├── model_training_all_in_one.py # Final, optimized model training script
├── streamlit_app.py            # Web application (Streamlit)
├── utils.py                    # Utility functions for the ensemble model
├── requirements.txt            # Dependencies
├── README.md                   # Documentation
├── data/                       # Dataset directory
│   ├── train/                # Training images
│   └── validation/           # Validation images
├── models/                     # Saved models
└── sample_images/              # Sample test images
```

### ⏮️ Prerequisites

Ensure you have the following installed:

  - Python 3.11+
  - Required libraries (can be installed via `requirements.txt`)

## 🛠️ Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/nikhilkr16/Plant-Disease-Detection-System.git
    ```
2.  Navigate to the repository:
    ```bash
    cd Plant-Disease-Detection-System
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download PlantVillage Dataset**
      - Download from [PlantVillage Dataset](https://www.kaggle.com/emmarex/plantdisease)
      - Extract and organize in the `data/` folder following the structure shown below

## 📊 Dataset Structure

```
data/
├── train/
│   ├── Apple_Apple_scab/
│   ├── Apple_Black_rot/
│   ├── Apple_Cedar_apple_rust/
│   ├── Apple_healthy/
│   └── Tomato_Bacterial_spot/
└── validation/
    ├── Apple_Apple_scab/
    ├── Apple_Black_rot/
    ├── Apple_Cedar_apple_rust/
    ├── Apple_healthy/
    └── Tomato_Bacterial_spot/
```

## 🏃‍♂️ Usage

### 1\. Train the Models

```bash
python model_training_all_in_one.py
```

This will:

  - Train multiple state-of-the-art models (e.g., EfficientNetV2, ResNet50V2) using a high-speed data pipeline.
  - Implement a two-stage fine-tuning strategy for each model.
  - Save the best version of each model (`.h5` file).
  - Generate and save training history and confusion matrix plots for each model.

### 2\. Run GUI Application

```bash
python main.py
```

Features:

  - Upload leaf images.
  - Get disease predictions from the powerful ensemble model.
  - View disease information and treatment suggestions.

### 3\. Run Web Application

```bash
streamlit run streamlit_app.py
```

Features:

  - Web-based interface with interactive charts.
  - Uses the combined power of the ensemble model for predictions.
  - Responsive design.

## 🧠 Model Architecture

  - **Strategy**: Transfer Learning & Fine-Tuning
  - **Base Models**: EfficientNetV2, ResNet50V2
  - **Classifier Head**: A custom head is added on top of each base model, consisting of:
      - `GlobalAveragePooling2D` to reduce feature dimensions.
      - `Dropout` for regularization to prevent overfitting.
      - A final `Dense` layer with `softmax` activation for 5-class classification.

## 📈 Training Details

  - **Data Pipeline**: High-performance `tf.data` API with `.cache()` and `.prefetch()` to maximize GPU utilization.
  - **Optimizer**: Adam optimizer. A higher learning rate is used for initial head training, and a very low learning rate (`1e-5`) is used for the fine-tuning stage.
  - **Training Strategy**: A two-stage process:
    1.  **Head Training:** Only the new classifier head is trained for 15 epochs.
    2.  **Fine-Tuning:** The top layers of the base model are unfrozen and trained along with the head for 35 more epochs.
  - **Callbacks**: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau.

## 🎯 Performance

The use of state-of-the-art models with a fine-tuning strategy achieves high accuracy on the validation data. Training history and confusion matrix plots are saved for each model to analyze its specific performance.

## 📱 Applications

  - **Agricultural Extension**: Help farmers identify diseases
  - **Research**: Support plant pathology research
  - **Education**: Teaching tool for agricultural students
  - **Mobile Apps**: Integration into farming applications

## Project Outputs

The system provides the following outputs:

1.  **Prediction Results**

      - Predicted disease class based on the averaged ensemble prediction.
      - Final confidence score.
      - A ranked list of all possible diseases and their confidence levels.

2.  **Disease Information**

      - Detailed description of the predicted disease.
      - Common symptoms.
      - Recommended treatment methods.

3.  **Visual Outputs**

      - Interactive confidence chart in the Streamlit app.
      - Static confidence chart in the Tkinter app.

4.  **Model Training Outputs** (for each model trained)

      - Training history plot (`_training_history.png`) showing accuracy and loss.
      - Confusion matrix plot (`_confusion_matrix.png`).
      - A saved model file (`_best.h5`).

5.  **Demo Mode Outputs** (when models are not available)

      - Mock predictions for demonstration purposes.

## 🔧 Technical Requirements

  - Python 3.11+
  - TensorFlow 2.15+
  - Streamlit, OpenCV, Scikit-learn
  - GPU for training is highly recommended.

## Tech Stack

  - **Python**: The primary programming language used for developing the application.
  - **TensorFlow/Keras**: Used for building and training the deep learning models.
  - **OpenCV**: Utilized for image processing tasks.
  - **Streamlit**: For creating the web application interface.
  - **Tkinter**: Used for building the GUI application.
  - **Pandas & NumPy**: For data manipulation and numerical operations.
  - **Matplotlib & Seaborn**: For data visualization and plotting the confusion matrix.
  - **Scikit-learn**: For generating the classification report.

## 📸 Sample Usage

1.  **Upload Image**: Select a clear leaf image.
2.  **Predict**: Click predict to analyze the image.
3.  **Results**: View the ensemble prediction confidence and disease information.
4.  **Treatment**: Follow recommended treatment guidelines.

## 🚨 Important Notes

  - Ensure good image quality (clear, well-lit, focused).
  - Model works best with images similar to training data.
  - Always consult agricultural experts for serious disease issues.
  - This tool is for educational and research purposes.

## 🤝 Contributing

1.  Fork the repository
2.  Create feature branch (`git checkout -b feature/AmazingFeature`)
3.  Commit changes (`git commit -m 'Add AmazingFeature'`)
4.  Push to branch (`git push origin feature/AmazingFeature`)
5.  Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

  - PlantVillage dataset creators
  - TensorFlow and Keras teams
  - Streamlit developers
  - Agricultural research community

## 📞 Contact

NIKHIL KUMAR
BTECH/10883/22
