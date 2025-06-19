import streamlit as st
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from PIL import Image

# Set page config
st.set_page_config(
    page_title="Iris Recognition System",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def apply_custom_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #4da6ff;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.8rem;
        color: #4da6ff;
        margin-bottom: 1rem;
    }
    .description {
        font-size: 1.1rem;
        margin-bottom: 1rem;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        color: #155724;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .warning-box {
        padding: 1rem;
        background-color: #fff3cd;
        color: #856404;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .error-box {
        padding: 1rem;
        background-color: #f8d7da;
        color: #721c24;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .step-box {
        padding: 1rem;
        background-color: #e2f0fd;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

def sidebar_menu():
    st.sidebar.title("Navigation")
    menu = st.sidebar.radio("Go to", ["Home", "Maintenance Mode", "About"])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("## System Status")
    
    with st.sidebar.expander("Dependencies"):
        try:
            import tensorflow as tf
            st.success(f"TensorFlow {tf.__version__}")
        except:
            st.error("TensorFlow not available")
            
        try:
            import cv2
            st.success(f"OpenCV {cv2.__version__}")
        except:
            st.error("OpenCV not available")
    
    return menu

def show_home():
    st.markdown('<h1 class="main-header">Iris Recognition System</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown('<p class="description">Welcome to the Iris Recognition System. This application demonstrates biometric identification using iris patterns.</p>', unsafe_allow_html=True)
        
        st.markdown('<div class="step-box"><h3>How It Works</h3><ol><li>Upload an eye image</li><li>The system extracts the iris region</li><li>Features are extracted using various techniques</li><li>The system matches against known iris patterns</li></ol></div>', unsafe_allow_html=True)
        
    with col2:
        st.image("https://upload.wikimedia.org/wikipedia/commons/2/27/Iris_-_left_eye_of_a_girl.jpg", caption="Sample Iris Image", width=300)
    
    st.markdown('<div class="warning-box">⚠️ <strong>Maintenance Mode:</strong> The system is currently in maintenance mode. Some features may be limited.</div>', unsafe_allow_html=True)

def show_maintenance_mode():
    st.markdown('<h1 class="main-header">Maintenance Mode</h1>', unsafe_allow_html=True)
    
    st.markdown('<div class="warning-box">⚠️ <strong>System Status:</strong> The application is currently in maintenance mode while we update dependencies.</div>', unsafe_allow_html=True)
    
    st.markdown("### System Information")
    
    # Display system information
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Python Environment")
        st.code(f"""
Python version: {sys.version}
Working directory: {os.getcwd()}
        """)
    
    with col2:
        st.markdown("#### Dependency Status")
        
        # Check for key libraries
        dependencies = [
            ("tensorflow", "TensorFlow (ML Framework)"),
            ("keras", "Keras (Neural Networks)"),
            ("cv2", "OpenCV (Computer Vision)"),
            ("pywt", "PyWavelets (Feature Extraction)")
        ]
        
        for module_name, description in dependencies:
            try:
                module = __import__(module_name)
                if hasattr(module, "__version__"):
                    st.success(f"✅ {description}: v{module.__version__}")
                else:
                    st.success(f"✅ {description}: Available")
            except ImportError:
                st.error(f"❌ {description}: Not available")
    
    # Display filesystem information
    st.markdown("### File System")
    
    try:
        # List top-level directories
        files = os.listdir(".")
        directories = [f for f in files if os.path.isdir(f)]
        files = [f for f in files if os.path.isfile(f)]
        
        st.write("Directories:")
        st.code("\n".join(directories))
        
        st.write("Files:")
        st.code("\n".join(files))
    except Exception as e:
        st.error(f"Error listing files: {str(e)}")

def show_about():
    st.markdown('<h1 class="main-header">About the Iris Recognition System</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Overview
    
    This iris recognition system demonstrates biometric identification using iris patterns. The system uses computer vision and machine learning techniques to extract features from iris images and compare them against known patterns.
    
    ### Technologies Used
    
    - **Computer Vision**: OpenCV for image processing and iris segmentation
    - **Machine Learning**: TensorFlow/Keras for deep learning models
    - **Feature Extraction**:
        - Convolutional Neural Networks
        - Gabor Filters for texture analysis
        - Wavelet Transform for frequency domain analysis
    - **Web Interface**: Streamlit for interactive web application
    
    ### Contact
    
    For more information, please contact the development team.
    """)

# Main app function
def main():
    apply_custom_css()
    
    # Initialize session state for navigation
    if "menu" not in st.session_state:
        st.session_state.menu = "Home"
    
    # Get menu selection from sidebar
    menu = sidebar_menu()
    
    # Update session state if menu changed
    if menu != st.session_state.menu:
        st.session_state.menu = menu
    
    # Show the appropriate page based on navigation
    if st.session_state.menu == "Home":
        show_home()
    elif st.session_state.menu == "Maintenance Mode":
        show_maintenance_mode()
    elif st.session_state.menu == "About":
        show_about()

# Run the app
if __name__ == "__main__":
    main()
