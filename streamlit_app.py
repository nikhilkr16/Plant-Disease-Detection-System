import streamlit as st
import numpy as np
from PIL import Image
import plotly.express as px
import pandas as pd
import os

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    from utils import EnsemblePredictor
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Your original Custom CSS is unchanged
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #2e7d32;
    text-align: center;
    margin-bottom: 2rem;
}
.prediction-box {
    background-color: #f8f9fa;
    border-radius: 10px;
    padding: 1rem;
    border-left: 5px solid #4CAF50;
    color:green;
}
.disease-info {
    background-color: #e3f2fd;
    border-radius: 10px;
    padding: 1rem;
    border-left: 5px solid #2196F3;
    color:#0d47a1;
}
.confidence-high {
    color: #4CAF50;
    font-weight: bold;
}
.confidence-medium {
    color: #FF9800;
    font-weight: bold;
}
.confidence-low {
    color: #F44336;
    font-weight: bold;
}
p{
            text-color: #000000;
            }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_predictor():
    """Load the ensemble predictor model or a mock version."""
    if TENSORFLOW_AVAILABLE:
        try:
            model_paths = [
                'models/efficientnetv2_best.h5',
                'models/resnet50v2_best.h5'
            ]
            return EnsemblePredictor(model_paths)
        except Exception as e:
            # If models fail to load, we'll fall through to the MockPredictor
            print(f"Failed to load models: {e}. Running in demo mode.")
            pass
    
    # If TF is not available or model loading fails, create and return a Mock Predictor
    class MockPredictor:
        def predict(self, pil_image):
            return {
                'predicted_class': 'Apple_Healthy (DEMO)',
                'confidence': 0.85,
                'all_predictions': { 'Apple_Healthy (DEMO)': 0.85, 'Apple_Scab (DEMO)': 0.10 }
            }
        def get_disease_info(self, disease_class):
            return {
                'description': 'This is a demo. Models or main libraries not found.',
                'symptoms': 'N/A',
                'treatment': 'N/A'
            }
    return MockPredictor()

def main():
    st.markdown('<h1 class="main-header">🌱 Plant Disease Detection System</h1>', 
                unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("📋 Instructions")
        st.markdown("""
        1. Upload a clear image of a plant leaf
        2. Ensure good lighting and focus
        3. Click 'Predict Disease' to analyze
        4. Review the results and recommendations
        """)
        
        if not TENSORFLOW_AVAILABLE:
            st.warning("⚠️ Running in DEMO mode. Main libraries not found.")
        
        st.header("🌿 Supported Plants")
        st.markdown("""
        - **Apple**: Scab, Black rot, Cedar rust
        - **Tomato**: Bacterial spot
        - **Healthy plants**: All types
        """)
    
    predictor = load_predictor()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📷 Image Upload")
        
        uploaded_file = st.file_uploader(
            "Choose a plant leaf image...",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            accept_multiple_files=False
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            st.info(f"📊 Image Size: {image.size[0]} x {image.size[1]} pixels")
            
            if st.button("🔍 Predict Disease", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    try:
                        result = predictor.predict(pil_image=image)
                        st.session_state.prediction_result = result
                        st.success("✅ Analysis complete!")
                    except Exception as e:
                        st.error(f"❌ Prediction failed: {e}")
        else:
            st.info("Please upload an image to begin analysis.")
    
    with col2:
        st.header("📊 Results")
        
        if 'prediction_result' in st.session_state:
            result = st.session_state.prediction_result
            predicted_class = result['predicted_class']
            confidence = result['confidence']
            
            st.markdown(f"""
            <div class="prediction-box">
                <h3>🏆 Predicted Disease</h3>
                <h2>{predicted_class.replace('_', ' ')}</h2>
                <p><strong>Confidence:</strong> {confidence:.2%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if confidence > 0.8:
                st.markdown('<p class="confidence-high">✅ High Confidence - Very likely correct</p>', 
                           unsafe_allow_html=True)
            elif confidence > 0.6:
                st.markdown('<p class="confidence-medium">⚠️ Moderate Confidence - Likely correct</p>', 
                           unsafe_allow_html=True)
            else:
                st.markdown('<p class="confidence-low">❗ Low Confidence - Consider expert consultation</p>', 
                           unsafe_allow_html=True)
            
            st.subheader("📈 All Predictions")
            all_preds = result['all_predictions']
            df = pd.DataFrame([
                {'Disease': k.replace('_', ' '), 'Confidence': v * 100} 
                for k, v in all_preds.items()
            ]).sort_values('Confidence', ascending=True)
            
            fig = px.bar(df, x='Confidence', y='Disease', orientation='h', title='Confidence Scores for All Classes', color='Confidence', color_continuous_scale='Viridis')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            disease_info = predictor.get_disease_info(predicted_class)
            st.markdown(f"""
            <div class="disease-info">
                <h3>🔬 Disease Information</h3>
                <p><strong>Description:</strong> {disease_info['description']}</p>
                <p><strong>Symptoms:</strong> {disease_info['symptoms']}</p>
                <p><strong>Treatment:</strong> {disease_info['treatment']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("📋 Detailed Results"):
                st.json(result)
        
        else:
            st.info("Upload an image and click 'Predict Disease' to see results here.")
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
            <p>🧑‍💻 Build By Nikhil Kumar | BTECH/10883/22 | BTECH: BIOTECHNOLOGY</p>
            <p>🏫 BIRLA INSTITUTE OF TECHNOLOGY MESRA, RANCHI ,JHARKHAND</p>
            <p>🌱 Plant Disease Detection System | Built with Streamlit</p>
        <p>For educational and research purposes. Always consult agricultural experts for serious issues.</p>
    </div>
    """, unsafe_allow_html=True)

    
def show_model_info():
    st.header("🧠 Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Architecture")
        st.markdown("""
        - **Strategy**: Transfer Learning & Fine-Tuning
        - **Base Models**: EfficientNetV2, ResNet50V2
        - **Input Size**: 224 x 224 x 3
        - **Classifier Head**: GlobalAveragePooling2D -> Dropout -> Dense
        - **Output**: 5 disease classes
        """)
    
    with col2:
        st.subheader("Training Details")
        st.markdown("""
        - **Dataset**: PlantVillage
        - **Data Pipeline**: High-Performance `tf.data` with `.cache()` and `.prefetch()`
        - **Optimizer**: Adam with a low learning rate for fine-tuning
        - **Training**: Two-stage fine-tuning (Head training then full model)
        - **Augmentation**: Keras Preprocessing Layers (RandomFlip, RandomRotation)
        """)
    
    st.markdown("---")
    st.subheader("📊 Model Performance")

    # Create columns for each model's results
    model_col1, model_col2 = st.columns(2)

    with model_col1:
        st.markdown("#### EfficientNetV2 Results")
        # Check for and display EfficientNetV2's specific output files
        if os.path.exists('models/efficientnetv2_training_history.png'):
            st.image('models/efficientnetv2_training_history.png', caption="Training History")
        
        if os.path.exists('models/efficientnetv2_confusion_matrix.png'):
            st.image('models/efficientnetv2_confusion_matrix.png', caption="Confusion Matrix")

    with model_col2:
        st.markdown("#### ResNet50V2 Results")
        # Check for and display ResNet50V2's specific output files
        if os.path.exists('models/resnet50v2_training_history.png'):
            st.image('models/resnet50v2_training_history.png', caption="Training History")
        
        if os.path.exists('models/resnet50v2_confusion_matrix.png'):
            st.image('models/resnet50v2_confusion_matrix.png', caption="Confusion Matrix")

def show_dataset_info():
    st.header("📊 Dataset Information")
    
    st.markdown("""
    ### PlantVillage Dataset
    
    The PlantVillage dataset is a collection of leaf images used for plant disease classification:
    
    - **Total Images**: 50,000+ images
    - **Plants**: 14 crop species
    - **Diseases**: 26 diseases + healthy plants
    - **Image Format**: RGB color images
    - **Resolution**: Various sizes (resized to 224x224 for training)
    
    ### Classes in This Model
    
    This model is trained on 5 classes:
    1. **Apple Apple Scab** - Fungal disease causing dark lesions
    2. **Apple Black Rot** - Serious fungal disease with brown/black lesions
    3. **Apple Cedar Apple Rust** - Disease causing yellow spots and orange lesions
    4. **Apple Healthy** - Healthy apple plants with no disease
    5. **Tomato Bacterial Spot** - Bacterial disease causing small dark spots
    
    
    ### IMAGES FOR  MODEL TAKEN FROM PLANTVILLAGE DATASET :

    
    This model is trained on 5 classes:
    1. **TRAINING-** - 4242 images.
    2. **VALIDATION-** - 1060 images.
    4. **TOTAL-** - 5302 images.




     """        
                )


def show_navigation():
    pages = {
        "🏠 Home": main,
        "🧠 Model Info": show_model_info,
        "📊 Dataset Info": show_dataset_info
    }
    with st.sidebar:
        st.markdown("---")
        st.header("Navigation")
        selected_page = st.radio("Go to", list(pages.keys()))
    pages[selected_page]()

if __name__ == "__main__":
    show_navigation()