#!/usr/bin/env python3
"""
Plant Disease Detection Demo Script (Updated for Ensemble Model)
This script demonstrates the functionality of the optimized plant disease detection system.
"""

import os
import time
from PIL import Image
# --- CHANGE: Import the new EnsemblePredictor ---
from utils import EnsemblePredictor
import matplotlib.pyplot as plt
import numpy as np

def print_banner():
    """Print system banner"""
    print("=" * 60)
    print("🌱 PLANT DISEASE DETECTION SYSTEM DEMO (ENSEMBLE) 🌱")
    print("=" * 60)
    print("AI-Powered Plant Health Analysis with SOTA Models")
    print("Built with TensorFlow")
    print("=" * 60)
    print()

def check_system_requirements():
    """Check if all system components are available"""
    print("📋 Checking System Requirements...")
    
    # --- CHANGE: Check for the new ensemble model files ---
    requirements = {
        "EfficientNetV2 Model": "models/efficientnetv2_best.h5",
        "ResNet50V2 Model": "models/resnet50v2_best.h5",
        "Class Names": "models/class_names.json",
    }
    
    all_good = True
    for name, path in requirements.items():
        if os.path.exists(path):
            print(f"✅ {name}: Found")
        else:
            print(f"❌ {name}: Missing ({path})")
            all_good = False
    
    if not all_good:
        print("\n⚠️  Some components are missing!")
        # --- CHANGE: Point to the new training script ---
        print("Run the following command to train the models:")
        print("1. python model_training_all_in_one.py")
        return False
    
    print("✅ All system requirements met!")
    return True

def demo_prediction_system():
    """Demonstrate the prediction system"""
    print("\n🔍 ENSEMBLE PREDICTION SYSTEM DEMO")
    print("-" * 30)
    
    try:
        # --- CHANGE: Initialize the EnsemblePredictor ---
        model_paths = [
            'models/efficientnetv2_best.h5',
            'models/resnet50v2_best.h5'
        ]
        predictor = EnsemblePredictor(model_paths)
        print("✅ Ensemble models loaded successfully!")
        print(f"📊 Models support {len(predictor.class_names)} classes:")
        
        for i, class_name in enumerate(predictor.class_names, 1):
            print(f"   {i}. {class_name.replace('_', ' ')}")
            
    except Exception as e:
        print(f"❌ Error loading predictor: {e}")
        return None
    
    return predictor

def run_sample_predictions(predictor):
    """Run predictions on sample images"""
    print("\n🧪 RUNNING SAMPLE PREDICTIONS")
    print("-" * 30)
    
    sample_dir = 'sample_images'
    if not os.path.exists(sample_dir):
        print("❌ No sample images found! Please create a 'sample_images' folder.")
        return []
        
    sample_files = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    if not sample_files:
        print("❌ No sample images found in the folder!")
        return []
    
    results = []
    # Test up to 3 images
    for i, image_file in enumerate(sample_files[:3], 1):
        image_path = os.path.join(sample_dir, image_file)
        print(f"🖼️  Testing Image {i}: {image_file}")
        
        try:
            # --- CHANGE: Open image with PIL and call the new predict method ---
            image = Image.open(image_path)
            result = predictor.predict(pil_image=image)
            
            predicted_class = result['predicted_class']
            confidence = result['confidence']
            
            print(f"   🏆 Prediction: {predicted_class.replace('_', ' ')}")
            print(f"   📊 Confidence: {confidence:.2%}")
            
            disease_info = predictor.get_disease_info(predicted_class)
            print(f"   🔬 Info: {disease_info.get('description', 'N/A')[:50]}...")
            
            results.append(result)
            print()
        except Exception as e:
            print(f"   ❌ Prediction failed for {image_file}: {e}")
            continue
    
    return results

def display_system_stats():
    """Display system statistics"""
    print("\n📊 SYSTEM STATISTICS")
    print("-" * 30)
    
    # --- CHANGE: Check sizes of all models ---
    model_files = ['models/efficientnetv2_best.h5', 'models/resnet50v2_best.h5']
    total_size = 0
    for model_file in model_files:
        if os.path.exists(model_file):
            model_size_mb = os.path.getsize(model_file) / (1024 * 1024)
            print(f"🧠 {os.path.basename(model_file)} Size: {model_size_mb:.2f} MB")
            total_size += model_size_mb
    
    print(f"🧠 Total Model Size: {total_size:.2f} MB")

def main():
    """Main demo function"""
    print_banner()
    
    if not check_system_requirements():
        return
    
    predictor = demo_prediction_system()
    if not predictor:
        return
    
    run_sample_predictions(predictor)
    display_system_stats()
    
    print("\n🎉 DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("Next Steps:")
    print("1. Run 'python main.py' for the desktop GUI")
    print("2. Run 'streamlit run streamlit_app.py' for the web app")
    print("=" * 60)

if __name__ == "__main__":
    main()
