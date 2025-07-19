#!/usr/bin/env python3
"""
Plant Disease Detection Demo Script
This script demonstrates the functionality of the plant disease detection system.
"""

import os
import sys
import time
from utils import PlantDiseasePredictor, create_sample_images
import matplotlib.pyplot as plt
import numpy as np

def print_banner():
    """Print system banner"""
    print("=" * 60)
    print("🌱 PLANT DISEASE DETECTION SYSTEM DEMO 🌱")
    print("=" * 60)
    print("AI-Powered Plant Health Analysis")
    print("Built with TensorFlow & OpenCV")
    print("=" * 60)
    print()

def check_system_requirements():
    """Check if all system components are available"""
    print("📋 Checking System Requirements...")
    
    requirements = {
        "Model File": "models/best_plant_disease_model.h5",
        "Class Names": "models/class_names.json",
        "Sample Images": "sample_images/"
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
        print("Run the following commands to set up:")
        print("1. python model_training.py  (to train the model)")
        print("2. python utils.py           (to create sample images)")
        return False
    
    print("✅ All system requirements met!")
    return True

def demo_prediction_system():
    """Demonstrate the prediction system"""
    print("\n🔍 PREDICTION SYSTEM DEMO")
    print("-" * 30)
    
    # Initialize predictor
    try:
        predictor = PlantDiseasePredictor()
        if predictor.model is None:
            print("❌ Failed to load model!")
            return False
        
        print("✅ Model loaded successfully!")
        print(f"📊 Model supports {len(predictor.class_names)} classes:")
        
        for i, class_name in enumerate(predictor.class_names, 1):
            print(f"   {i}. {class_name.replace('_', ' ')}")
        
    except Exception as e:
        print(f"❌ Error loading predictor: {e}")
        return False
    
    return predictor

def run_sample_predictions(predictor):
    """Run predictions on sample images"""
    print("\n🧪 RUNNING SAMPLE PREDICTIONS")
    print("-" * 30)
    
    # Create sample images if they don't exist
    if not os.path.exists('sample_images'):
        print("Creating sample images...")
        create_sample_images()
    
    sample_dir = 'sample_images'
    sample_files = [f for f in os.listdir(sample_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    if not sample_files:
        print("❌ No sample images found!")
        return
    
    print(f"Found {len(sample_files)} sample images")
    print()
    
    results = []
    
    for i, image_file in enumerate(sample_files[:3], 1):  # Test first 3 images
        image_path = os.path.join(sample_dir, image_file)
        print(f"🖼️  Testing Image {i}: {image_file}")
        
        # Make prediction
        result, message = predictor.predict_disease(image_path=image_path)
        
        if result is None:
            print(f"   ❌ Prediction failed: {message}")
            continue
        
        # Display results
        predicted_class = result['predicted_class']
        confidence = result['confidence']
        
        print(f"   🏆 Prediction: {predicted_class.replace('_', ' ')}")
        print(f"   📊 Confidence: {confidence:.2%}")
        
        # Show top 3 predictions
        print("   📈 Top 3 Predictions:")
        for j, (class_name, conf) in enumerate(result['top_3_predictions'], 1):
            print(f"      {j}. {class_name.replace('_', ' ')}: {conf:.2%}")
        
        # Get disease info
        disease_info = predictor.get_disease_info(predicted_class)
        print(f"   🔬 Info: {disease_info['description'][:50]}...")
        
        results.append(result)
        print()
    
    return results

def display_system_stats():
    """Display system statistics"""
    print("\n📊 SYSTEM STATISTICS")
    print("-" * 30)
    
    # Check model file size
    if os.path.exists('models/best_plant_disease_model.h5'):
        model_size = os.path.getsize('models/best_plant_disease_model.h5') / (1024 * 1024)  # MB
        print(f"🧠 Model Size: {model_size:.2f} MB")
    
    # Count sample images
    if os.path.exists('sample_images'):
        sample_count = len([f for f in os.listdir('sample_images') if f.endswith(('.jpg', '.png', '.jpeg'))])
        print(f"🖼️  Sample Images: {sample_count}")
    
    # Check data directories
    if os.path.exists('data'):
        train_classes = len([d for d in os.listdir('data/train') if os.path.isdir(os.path.join('data/train', d))]) if os.path.exists('data/train') else 0
        val_classes = len([d for d in os.listdir('data/validation') if os.path.isdir(os.path.join('data/validation', d))]) if os.path.exists('data/validation') else 0
        print(f"📚 Training Classes: {train_classes}")
        print(f"📝 Validation Classes: {val_classes}")

def demo_gui_info():
    """Show information about GUI applications"""
    print("\n🖥️  GUI APPLICATIONS")
    print("-" * 30)
    
    print("1. 🖼️  Desktop GUI (tkinter):")
    print("   Command: python main.py")
    print("   Features: Image upload, prediction, charts, disease info")
    print()
    
    print("2. 🌐 Web App (Streamlit):")
    print("   Command: streamlit run streamlit_app.py")
    print("   Features: Web interface, drag-drop upload, interactive charts")
    print()
    
    print("3. 📱 API Integration:")
    print("   The prediction system can be integrated into mobile apps")
    print("   or other web services using the PlantDiseasePredictor class")

def create_demo_visualization(results):
    """Create visualization of demo results"""
    if not results:
        return
    
    print("\n📈 CREATING DEMO VISUALIZATION")
    print("-" * 30)
    
    try:
        # Extract confidence scores
        confidences = [result['confidence'] for result in results]
        predictions = [result['predicted_class'].replace('_', ' ') for result in results]
        
        # Create bar chart
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(predictions)), [c * 100 for c in confidences], 
                      color=['#4CAF50', '#2196F3', '#FF9800', '#F44336', '#9C27B0'][:len(predictions)])
        
        plt.title('Demo Prediction Confidence Scores', fontsize=16, fontweight='bold')
        plt.xlabel('Test Images', fontsize=12)
        plt.ylabel('Confidence (%)', fontsize=12)
        plt.xticks(range(len(predictions)), [f'Image {i+1}' for i in range(len(predictions))])
        plt.ylim(0, 100)
        
        # Add value labels on bars
        for i, (bar, conf, pred) in enumerate(zip(bars, confidences, predictions)):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{conf:.1%}\n{pred[:15]}...', 
                    ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('demo_results.png', dpi=300, bbox_inches='tight')
        print("✅ Visualization saved as 'demo_results.png'")
        
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")

def main():
    """Main demo function"""
    print_banner()
    
    # Check system requirements
    if not check_system_requirements():
        print("\n❌ Demo cannot proceed due to missing components.")
        print("Follow the setup instructions above and try again.")
        return
    
    # Initialize prediction system
    predictor = demo_prediction_system()
    if not predictor:
        return
    
    # Run sample predictions
    results = run_sample_predictions(predictor)
    
    # Display system stats
    display_system_stats()
    
    # Show GUI info
    demo_gui_info()
    
    # Create visualization
    if results:
        create_demo_visualization(results)
    
    print("\n🎉 DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("Next Steps:")
    print("1. Run 'python main.py' for desktop GUI")
    print("2. Run 'streamlit run streamlit_app.py' for web app")
    print("3. Upload your own plant images for testing")
    print("=" * 60)

if __name__ == "__main__":
    main() 