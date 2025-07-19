# 🌱 Plant Disease Detection System

An AI-powered system for detecting plant diseases from leaf images using Convolutional Neural Networks (CNN).

## 📋 Project Overview

This project uses deep learning to classify plant diseases from leaf images. The system is trained on the PlantVillage dataset and can identify 5 different classes:

- **Apple Apple Scab**
- **Apple Black Rot** 
- **Apple Cedar Apple Rust**
- **Apple Healthy**
- **Tomato Bacterial Spot**

## 🚀 Features

- **CNN Model**: Custom-built with TensorFlow/Keras
- **GUI Application**: User-friendly tkinter interface
- **Web Deployment**: Streamlit web application
- **High Accuracy**: Trained with data augmentation for robust performance
- **Disease Information**: Provides symptoms and treatment recommendations

## 📁 Project Structure

```
PROJECT/
├── main.py                 # GUI application (tkinter)
├── model_training.py       # Model training script
├── streamlit_app.py        # Web application (Streamlit)
├── utils.py               # Utility functions
├── requirements.txt       # Dependencies
├── README.md             # Documentation
├── data/                 # Dataset directory
│   ├── train/           # Training images
│   └── validation/      # Validation images
├── models/              # Saved models
└── sample_images/       # Sample test images
```



### ⏮️ Prerequisites

Ensure you have the following installed:

- Python 3.11
- Required libraries (can be installed via `requirements.txt`)

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/nikhilkr16/Plant-Disease-Detection-System.git
   ```
2. Navigate to the repository:
   ```bash
   cd Plant-Disease-Detection-System
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download PlantVillage Dataset**
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

### 1. Train the Model

```bash
python model_training.py
```

This will:
- Create data directory structure
- Train CNN model with data augmentation
- Save best model and training history
- Generate performance plots

### 2. Run GUI Application

```bash
python main.py
```

Features:
- Upload leaf images
- Get disease predictions with confidence scores
- View disease information and treatment suggestions
- Interactive charts showing prediction confidence

### 3. Run Web Application

```bash
streamlit run streamlit_app.py
```

Features:
- Web-based interface
- Drag-and-drop image upload
- Interactive confidence charts
- Responsive design

## 🧠 Model Architecture

- **Input Layer**: 224 x 224 x 3 (RGB images)
- **Conv2D Blocks**: 4 blocks with BatchNormalization
- **Filters**: 32, 64, 128, 256
- **Dense Layers**: 512 and 256 neurons with Dropout
- **Output Layer**: 5 classes with Softmax activation

## 📈 Training Details

- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: Categorical Crossentropy
- **Batch Size**: 32
- **Epochs**: 50 (with early stopping)
- **Data Augmentation**: Rotation, shifting, zoom, flip
- **Callbacks**: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

## 🎯 Performance

The model achieves high accuracy on validation data with robust performance across all disease classes. Training history and confusion matrix are saved for analysis.

## 📱 Applications

- **Agricultural Extension**: Help farmers identify diseases
- **Research**: Support plant pathology research
- **Education**: Teaching tool for agricultural students
- **Mobile Apps**: Integration into farming applications

## 🔧 Technical Requirements

- Python 3.7+
- TensorFlow 2.13+
- OpenCV 4.8+
- Streamlit 1.28+
- Sufficient GPU memory for training (optional but recommended)

## 📸 Sample Usage

1. **Upload Image**: Select a clear leaf image
2. **Predict**: Click predict to analyze the image
3. **Results**: View prediction confidence and disease information
4. **Treatment**: Follow recommended treatment guidelines

## 🚨 Important Notes

- Ensure good image quality (clear, well-lit, focused)
- Model works best with images similar to training data
- Always consult agricultural experts for serious disease issues
- This tool is for educational and research purposes

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- PlantVillage dataset creators
- TensorFlow and Keras teams
- Streamlit developers
- Agricultural research community

## 📞 Contact

For questions or support, please open an issue in the repository.

## Project Outputs

The system provides the following outputs:

1. **Prediction Results**
   - Predicted disease class
   - Confidence score
   - Top 3 predictions with confidence levels
   - Confidence level interpretation (High/Moderate/Low)

2. **Disease Information**
   - Detailed description of the predicted disease
   - Common symptoms
   - Recommended treatment methods

3. **Visual Outputs**
   - Interactive confidence chart showing top 5 predictions
   - Uploaded image display
   - Color-coded confidence indicators

4. **Model Training Outputs** (when training is performed)
   - Training history plots (accuracy and loss)
   - Model evaluation metrics
   - Confusion matrix
   - Saved model file (best_plant_disease_model.h5)

5. **Demo Mode Outputs** (when model is not available)
   - Mock predictions for demonstration
   - Sample disease information
   - Simulated confidence scores

## Technical Details

---

**Built with ❤️ for sustainable agriculture and AI education** # -Plant-Disease-Detection-System
