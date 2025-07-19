import os
import numpy as np
from PIL import Image
import random

# Try to import TensorFlow, else use mock mode
tf_available = True
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except ImportError:
    tf_available = False

class ImagePreprocessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
    
    def preprocess_image(self, image_path):
        """Preprocess image for model prediction"""
        try:
            from cv2 import imread, cvtColor, COLOR_BGR2RGB, resize
            img = imread(image_path)
            if img is None:
                raise ValueError("Could not load image")
            
            img = cvtColor(img, COLOR_BGR2RGB)
            img = resize(img, self.target_size)
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=0)
            return img
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
    
    def preprocess_pil_image(self, pil_image):
        """Preprocess PIL image for model prediction"""
        try:
            import cv2
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            img = np.array(pil_image)
            img = cv2.resize(img, self.target_size)
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=0)
            return img
        except Exception as e:
            print(f"Error preprocessing PIL image: {e}")
            return None

class PlantDiseasePredictor:
    def __init__(self, model_path='models/best_plant_disease_model.h5', 
                 class_names_path='models/class_names.json'):
        self.model_path = model_path
        self.class_names_path = class_names_path
        self.model = None
        self.class_names = [
            'Apple_Apple_scab',
            'Apple_Black_rot',
            'Apple_Cedar_apple_rust',
            'Apple_healthy',
            'Tomato_Bacterial_spot'
        ]
        self.preprocessor = ImagePreprocessor()
        self.mock_mode = not tf_available or not os.path.exists(model_path)
        if not self.mock_mode:
            self.load_model_and_classes()
        else:
            print("[INFO] Running in MOCK/DEMO mode: No real predictions will be made.")
    
    def load_model_and_classes(self):
        """Load trained model and class names"""
        try:
            self.model = load_model(self.model_path)
            import json
            with open(self.class_names_path, 'r') as f:
                self.class_names = json.load(f)
            print("Model and class names loaded.")
        except Exception as e:
            print(f"Error loading model or classes: {e}")
            self.mock_mode = True
    
    def predict_disease(self, image_path=None, pil_image=None):
        """Predict plant disease from image"""
        if self.mock_mode:
            # Return a random prediction for demo
            idx = random.randint(0, len(self.class_names)-1)
            confidence = round(random.uniform(0.6, 0.99), 2)
            top_3 = random.sample(list(enumerate(self.class_names)), 3)
            top_3_predictions = [(name, round(random.uniform(0.5, 0.99), 2)) for i, name in top_3]
            all_predictions = {name: round(random.uniform(0.01, 1.0), 2) for name in self.class_names}
            result = {
                'predicted_class': self.class_names[idx],
                'confidence': confidence,
                'top_3_predictions': top_3_predictions,
                'all_predictions': all_predictions
            }
            return result, "MOCK: Success"
        if self.model is None:
            return None, "Model not loaded"
        
        if image_path:
            processed_img = self.preprocessor.preprocess_image(image_path)
        elif pil_image:
            processed_img = self.preprocessor.preprocess_pil_image(pil_image)
        else:
            return None, "No image provided"
        
        if processed_img is None:
            return None, "Error preprocessing image"
        
        try:
            predictions = self.model.predict(processed_img)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class_idx]
            
            predicted_class = self.class_names[predicted_class_idx]
            
            top_3_idx = np.argsort(predictions[0])[-3:][::-1]
            top_3_predictions = []
            
            for idx in top_3_idx:
                class_name = self.class_names[idx]
                conf = predictions[0][idx]
                top_3_predictions.append((class_name, conf))
            
            result = {
                'predicted_class': predicted_class,
                'confidence': float(confidence),
                'top_3_predictions': top_3_predictions,
                'all_predictions': {self.class_names[i]: float(predictions[0][i]) 
                                 for i in range(len(self.class_names))}
            }
            
            return result, "Success"
        except Exception as e:
            return None, f"Error making prediction: {e}"
    
    def get_disease_info(self, disease_name):
        """Get information about the disease"""
        disease_info = {
            'Apple_Apple_scab': {
                'description': 'Apple scab is a fungal disease that affects apple trees.',
                'symptoms': 'Dark, scaly lesions on leaves and fruit.',
                'treatment': 'Apply fungicide sprays and ensure proper air circulation.'
            },
            'Apple_Black_rot': {
                'description': 'Black rot is a serious disease of apples caused by fungi.',
                'symptoms': 'Brown to black lesions on leaves, fruit rot.',
                'treatment': 'Remove infected plant parts and apply appropriate fungicides.'
            },
            'Apple_Cedar_apple_rust': {
                'description': 'Cedar apple rust is a fungal disease affecting apple trees.',
                'symptoms': 'Yellow spots on leaves, orange lesions.',
                'treatment': 'Remove nearby cedar trees and apply preventive fungicides.'
            },
            'Apple_healthy': {
                'description': 'Healthy apple plant with no signs of disease.',
                'symptoms': 'Green, healthy leaves with no lesions or discoloration.',
                'treatment': 'Continue regular care and monitoring.'
            },
            'Tomato_Bacterial_spot': {
                'description': 'Bacterial spot is a common disease affecting tomato plants.',
                'symptoms': 'Small, dark spots on leaves and fruit.',
                'treatment': 'Use copper-based bactericides and ensure proper plant spacing.'
            }
        }
        
        return disease_info.get(disease_name, {
            'description': 'Information not available for this disease.',
            'symptoms': 'Please consult agricultural experts.',
            'treatment': 'Seek professional advice for treatment options.'
        })

def create_sample_images():
    """Create sample placeholder images for testing"""
    import cv2
    import os
    
    os.makedirs('sample_images', exist_ok=True)
    
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    disease_names = ['Apple_Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 
                    'Apple_healthy', 'Tomato_Bacterial_spot']
    
    for i, (color, name) in enumerate(zip(colors, disease_names)):
        img = np.full((224, 224, 3), color, dtype=np.uint8)
        noise = np.random.randint(0, 50, (224, 224, 3))
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        cv2.imwrite(f'sample_images/sample_{name}.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    print("Sample images created in 'sample_images' folder")

if __name__ == "__main__":
    create_sample_images() 