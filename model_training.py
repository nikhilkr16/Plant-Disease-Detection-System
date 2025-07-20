import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import cv2
from PIL import Image
import requests
import zipfile
import shutil

class PlantDiseaseDetector:
    def __init__(self, img_height=224, img_width=224, batch_size=32):
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.model = None
        self.class_names = []
        
    def download_sample_data(self):
        """Download and prepare sample plant disease data"""
        print("Setting up sample data structure...")
        pass
        
        # Create data directories
        os.makedirs('data/train', exist_ok=True)
        os.makedirs('data/validation', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        
        # Create sample disease classes
        disease_classes = [
            'Apple_Apple_scab',
            'Apple_Black_rot',
            'Apple_Cedar_apple_rust',
            'Apple_healthy',
            'Tomato_Bacterial_spot'
        ]
        
        for class_name in disease_classes:
            os.makedirs(f'data/train/{class_name}', exist_ok=True)
            os.makedirs(f'data/validation/{class_name}', exist_ok=True)
        
        print("Data structure created. Please add your PlantVillage dataset images to the respective folders.")
        print("Expected structure:")
        print("data/")
        print("├── train/")
        print("│   ├── Apple_Apple_scab/")
        print("│   ├── Apple_Black_rot/")
        print("│   ├── Apple_Cedar_apple_rust/")
        print("│   ├── Apple_healthy/")
        print("│   └── Tomato_Bacterial_spot/")
        print("└── validation/")
        print("    ├── Apple_Apple_scab/")
        print("    ├── Apple_Black_rot/")
        print("    ├── Apple_Cedar_apple_rust/")
        print("    ├── Apple_healthy/")
        print("    └── Tomato_Bacterial_spot/")
        
    def prepare_data_generators(self):
        """Prepare data generators with augmentation"""
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=50,
            width_shift_range=0.3,
            height_shift_range=0.3,
            shear_range=0.3,
            zoom_range=0.3,
            vertical_flip=True,     
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Only rescaling for validation
        validation_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            'data/train',
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical'
        )
        
        validation_generator = validation_datagen.flow_from_directory(
            'data/validation',
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical'
        )
        
        self.class_names = list(train_generator.class_indices.keys())
        print(f"Classes found: {self.class_names}")
        
        return train_generator, validation_generator
    
    # def build_model(self, num_classes):
        # """Build CNN model architecture"""
        # model = Sequential([
        #     # First Convolutional Block
        #     Conv2D(32, (3, 3), activation='relu', input_shape=(self.img_height, self.img_width, 3)),
        #     BatchNormalization(),
        #     MaxPooling2D(2, 2),
            
        #     # Second Convolutional Block
        #     Conv2D(64, (3, 3), activation='relu'),
        #     BatchNormalization(),
        #     MaxPooling2D(2, 2),
            
        #     # Third Convolutional Block
        #     Conv2D(128, (3, 3), activation='relu'),
        #     BatchNormalization(),
        #     MaxPooling2D(2, 2),
            
        #     # Fourth Convolutional Block
        #     Conv2D(256, (3, 3), activation='relu'),
        #     BatchNormalization(),
        #     MaxPooling2D(2, 2),
            
        #     # Flatten and Dense layers
        #     Flatten(),
        #     Dense(512, activation='relu'),
        #     Dropout(0.5),
        #     Dense(256, activation='relu'),
        #     Dropout(0.3),
        #     Dense(num_classes, activation='softmax')
        # ])
        
    def build_vgg16_model(self, num_classes):
        """Build a transfer learning model using VGG16"""


        base_model = VGG16(input_shape=(self.img_height, self.img_width, 3),
                           include_top=False,
                           weights='imagenet')
        base_model.trainable = False

        x = Flatten()(base_model.output)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        output_layer = Dense(num_classes, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=output_layer)

        # Load the VGG16 model with pre-trained ImageNet weights,
        # excluding the final classification layer.
        base_model = VGG16(input_shape=(self.img_height, self.img_width, 3),
                           include_top=False,
                           weights='imagenet')

        # Freeze the layers of the base VGG16 model so they are not updated
        # during the training process.
        base_model.trainable = False

        # Create a new classifier head on top of the base model
        x = Flatten()(base_model.output)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        # The final output layer with softmax activation for multi-class classification
        output_layer = Dense(num_classes, activation='softmax')(x)

        # Combine the base model and the new classifier head
        model = Model(inputs=base_model.input, outputs=output_layer)

        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )


        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train_model(self, train_generator, validation_generator, epochs=200, class_weights=None):
        """Train the model"""
        
        history = self.model.fit(
            train_generator,
            class_weight=class_weights,
            # ... other parameters
        )

        # Callbacks
        checkpoint = ModelCheckpoint(
            'models/best_plant_disease_model.h5',
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        )
        
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.0001,
            verbose=1
        )
        
        # Calculate steps
        steps_per_epoch = train_generator.samples // self.batch_size
        validation_steps = validation_generator.samples // self.batch_size
        
        # Train model
        history = self.model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            callbacks=[checkpoint, early_stopping, reduce_lr],
            class_weight=class_weights,
            verbose=1
        )
        
        return history
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('models/training_history.png')
        plt.show()
        pass
    
    def evaluate_model(self, validation_generator):
        """Evaluate model performance"""
        # Load best model
        from tensorflow.keras.models import load_model
        self.model = load_model('models/best_plant_disease_model.h5')
        
        # Get predictions
        validation_generator.reset()
        predictions = self.model.predict(validation_generator, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Get true labels
        true_classes = validation_generator.classes
        class_labels = list(validation_generator.class_indices.keys())
        
        # Classification report
        report = classification_report(true_classes, predicted_classes, target_names=class_labels)
        print("Classification Report:")
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('models/confusion_matrix.png')
        plt.show()
        
        return report, cm
    pass
def main():
    # Initialize detector
    detector = PlantDiseaseDetector()
    
    # Setup data structure
    detector.download_sample_data()
    
    # Check if data exists
    if not os.path.exists('data/train') or len(os.listdir('data/train')) == 0:
        print("Please add your PlantVillage dataset images to the data folders before training.")
        return
    
    # Prepare data
    train_gen, val_gen = detector.prepare_data_generators()
    



    # Count the number of samples in each class
    class_counts = Counter(train_gen.classes)
    total_samples = len(train_gen.classes)
    num_classes = len(class_counts)

    # Calculate class weights
    class_weights = {}
    for i, count in class_counts.items():
        weight = total_samples / (num_classes * count)
        class_weights[i] = weight

    print("Calculated Class Weights:")
    for i, class_name in enumerate(detector.class_names):
        print(f"  - {class_name}: {class_weights[i]:.2f}")




    # Build model
    num_classes = len(detector.class_names)
    model = detector.build_vgg16_model(num_classes) 
        # model = detector.build_model(num_classes)
    
    print("Model Architecture:")
    model.summary()
    
    # Train model
    print("Starting training...")
    history = detector.train_model(train_gen, val_gen, epochs=200,class_weights=class_weights)
    
    # Plot history
    detector.plot_training_history(history)
    
    # Evaluate model
    detector.evaluate_model(val_gen)
    
    # Save class names
    import json
    with open('models/class_names.json', 'w') as f:
        json.dump(detector.class_names, f)
    
    print("Training completed! Model saved as 'models/best_plant_disease_model.h5'")

if __name__ == "__main__":
    main() 