# FINAL utils.py (Validation Removed)

import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as effnet_preprocess

class EnsemblePredictor:
    def __init__(self, model_paths, class_names_path='models/class_names.json'):
        for path in model_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found at {path}.")
        self.models = [load_model(path) for path in model_paths]
        print(f"Loaded {len(self.models)} models for the ensemble.")

        with open(class_names_path, 'r') as f:
            self.class_names = json.load(f)

        self.preprocess_funcs = [effnet_preprocess, resnet_preprocess]

    def predict(self, pil_image, img_size=(224, 224)):
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        pil_image = pil_image.resize(img_size)
        img_array = image.img_to_array(pil_image)
        
        all_preds = []
        for model, preprocess_func in zip(self.models, self.preprocess_funcs):
            processed_img = np.expand_dims(img_array.copy(), axis=0)
            processed_img = preprocess_func(processed_img)
            preds = model.predict(processed_img)
            all_preds.append(preds)
            
        avg_preds = np.mean(all_preds, axis=0)
        class_idx = np.argmax(avg_preds[0])
        predicted_class = self.class_names[class_idx]
        confidence = avg_preds[0][class_idx]
        all_predictions_dict = {self.class_names[i]: float(avg_preds[0][i]) for i in range(len(self.class_names))}

        return {'predicted_class': predicted_class, 'confidence': confidence, 'all_predictions': all_predictions_dict}

    def get_disease_info(self, disease_name):
        disease_info = {
            'Apple_Apple_scab': {'description': 'Apple scab is a fungal disease...', 'symptoms': 'Dark, scaly lesions...', 'treatment': 'Apply fungicide...'},
            'Apple_Black_rot': {'description': 'Black rot is a serious disease...', 'symptoms': 'Brown to black lesions...', 'treatment': 'Remove infected parts...'},
            'Apple_Cedar_apple_rust': {'description': 'Cedar apple rust is a fungal disease...', 'symptoms': 'Yellow spots...', 'treatment': 'Remove nearby cedar trees...'},
            'Apple_healthy': {'description': 'Healthy apple plant.', 'symptoms': 'No lesions or discoloration.', 'treatment': 'Continue regular care.'},
            'Tomato_Bacterial_spot': {'description': 'Bacterial spot is a common disease...', 'symptoms': 'Small, dark spots...', 'treatment': 'Use copper-based bactericides.'}
        }
        return disease_info.get(disease_name, {'description': 'Info not available.'})