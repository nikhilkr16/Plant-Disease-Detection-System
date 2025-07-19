import streamlit as st
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os

# Try importing the predictor
try:
    from utils import PlantDiseasePredictor
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    st.warning("⚠️ Running in DEMO mode: No real predictions will be made.")

# Page configuration
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
    """Load the plant disease predictor model"""
    if TENSORFLOW_AVAILABLE:
        return PlantDiseasePredictor()
    else:
        # Return a mock predictor for demo mode
        class MockPredictor:
            def predict_disease(self, pil_image):
                # Return mock predictions
                mock_result = {
                    'predicted_class': 'Apple_Healthy',
                    'confidence': 0.85,
                    'all_predictions': {
                        'Apple_Healthy': 0.85,
                        'Apple_Scab': 0.10,
                        'Apple_Black_Rot': 0.05
                    },
                    'top_3_predictions': [
                        ('Apple_Healthy', 0.85),
                        ('Apple_Scab', 0.10),
                        ('Apple_Black_Rot', 0.05)
                    ]
                }
                return mock_result, "Demo mode prediction"
            
            def get_disease_info(self, disease_class):
                return {
                    'description': 'This is a demo prediction. The actual model is not available in this environment.',
                    'symptoms': 'No real symptoms available in demo mode.',
                    'treatment': 'Please run the model locally for accurate predictions.'
                }
        return MockPredictor()

def main():
    # Title
    st.markdown('<h1 class="main-header">🌱 Plant Disease Detection System</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Instructions")
        st.markdown("""
        1. Upload a clear image of a plant leaf
        2. Ensure good lighting and focus
        3. Click 'Predict Disease' to analyze
        4. Review the results and recommendations
        """)
        
        if not TENSORFLOW_AVAILABLE:
            st.warning("⚠️ Running in DEMO mode")
            st.info("For real predictions, run the app locally with TensorFlow installed.")
        
        st.header("🌿 Supported Plants")
        st.markdown("""
        - **Apple**: Scab, Black rot, Cedar rust
        - **Tomato**: Bacterial spot
        - **Healthy plants**: All types
        """)
    
    # Load predictor
    predictor = load_predictor()
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📷 Image Upload")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a plant leaf image...",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            accept_multiple_files=False
        )
        
        if uploaded_file is not None:
            # Display image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Image info
            st.info(f"📊 Image Size: {image.size[0]} x {image.size[1]} pixels")
            
            # Predict button
            if st.button("🔍 Predict Disease", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    # Make prediction
                    result, message = predictor.predict_disease(pil_image=image)
                    
                    if result is None:
                        st.error(f"❌ Prediction failed: {message}")
                    else:
                        # Store result in session state
                        st.session_state.prediction_result = result
                        st.success("✅ Analysis complete!")
        else:
            st.info("Please upload an image to begin analysis.")
    
    with col2:
        st.header("📊 Results")
        
        if 'prediction_result' in st.session_state:
            result = st.session_state.prediction_result
            
            # Main prediction
            predicted_class = result['predicted_class']
            confidence = result['confidence']
            
            # Prediction box
            st.markdown(f"""
            <div class="prediction-box">
                <h3>🏆 Predicted Disease</h3>
                <h2>{predicted_class.replace('_', ' ')}</h2>
                <p><strong>Confidence:</strong> {confidence:.2%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence interpretation
            if confidence > 0.8:
                st.markdown('<p class="confidence-high">✅ High Confidence - Very likely correct</p>', 
                           unsafe_allow_html=True)
            elif confidence > 0.6:
                st.markdown('<p class="confidence-medium">⚠️ Moderate Confidence - Likely correct</p>', 
                           unsafe_allow_html=True)
            else:
                st.markdown('<p class="confidence-low">❗ Low Confidence - Consider expert consultation</p>', 
                           unsafe_allow_html=True)
            
            # Top predictions chart
            st.subheader("📈 All Predictions")
            
            # Prepare data for chart
            all_preds = result['all_predictions']
            df = pd.DataFrame([
                {'Disease': k.replace('_', ' '), 'Confidence': v * 100} 
                for k, v in all_preds.items()
            ]).sort_values('Confidence', ascending=True)
            
            # Create horizontal bar chart
            fig = px.bar(
                df, 
                x='Confidence', 
                y='Disease',
                orientation='h',
                title='Confidence Scores for All Classes',
                color='Confidence',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Disease information
            disease_info = predictor.get_disease_info(predicted_class)
            
            st.markdown(f"""
            <div class="disease-info">
                <h3>🔬 Disease Information</h3>
                <p><strong>Description:</strong> {disease_info['description']}</p>
                <p><strong>Symptoms:</strong> {disease_info['symptoms']}</p>
                <p><strong>Treatment:</strong> {disease_info['treatment']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Detailed results in expander
            with st.expander("📋 Detailed Results"):
                st.json(result)
        
        else:
            st.info("Upload an image and click 'Predict Disease' to see results here.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
                <p>🧑‍💻 Build By Nikhil Kumar | BTECH/10883/22 | BTECH: BIOTECHNOLOGY</p>
                <p>🏫 BIRLA INSTITUTE OF TECHNOLOGY MESRA, RANCHI ,JHARKHAND</p>
                <p>🌱 Plant Disease Detection System | Built with Streamlit</p>
            <p>For educational and research purposes. Always consult agricultural experts for serious issues.</p>
    </div>
    """, unsafe_allow_html=True)

# Additional pages
def show_model_info():
    st.header("🧠 Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Architecture")
        st.markdown("""
        - **Type**: Convolutional Neural Network (CNN)
        - **Input Size**: 224 x 224 x 3
        - **Layers**: 4 Conv2D blocks with BatchNorm
        - **Dense Layers**: 2 fully connected layers
        - **Output**: 5 disease classes
        """)
    
    with col2:
        st.subheader("Training Details")
        st.markdown("""
        - **Dataset**: PlantVillage
        - **Optimizer**: Adam
        - **Loss**: Categorical Crossentropy
        - **Metrics**: Accuracy
        - **Augmentation**: Rotation, Shift, Zoom, Flip
        """)
    
    # Model performance (if available)
    if os.path.exists('models/training_history.png'):
        st.subheader("📊 Training History")
        st.image('models/training_history.png')
    
    if os.path.exists('models/confusion_matrix.png'):
        st.subheader("🎯 Confusion Matrix")
        st.image('models/confusion_matrix.png')

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
    """)

# Navigation
def show_navigation():
    pages = {
        "🏠 Home": main,
        "🧠 Model Info": show_model_info,
        "📊 Dataset Info": show_dataset_info
    }
    
    # Page selection
    if 'page' not in st.session_state:
        st.session_state.page = "🏠 Home"
    
    # Navigation in sidebar
    with st.sidebar:
        st.markdown("---")
        selected_page = st.radio("Navigation", list(pages.keys()))
        
        if selected_page != st.session_state.page:
            st.session_state.page = selected_page
            st.rerun()
    
    # Show selected page
    pages[st.session_state.page]()

if __name__ == "__main__":
    if len(st.session_state) == 0:
        show_navigation()
    else:
        show_navigation() 