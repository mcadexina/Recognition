import streamlit as st
import os
import importlib
import sys

# Try to import TensorFlow
try:
    import tensorflow as tf
    print(f"TensorFlow {tf.__version__} found")
    # Try to import Keras
    try:
        from tensorflow.keras.models import load_model
        print("Using TensorFlow Keras")
    except:
        import keras
        print(f"Using standalone Keras {keras.__version__}")
except Exception as e:
    print(f"TensorFlow import error: {str(e)}")
    st.error(f"TensorFlow import error: {str(e)}")

# Try to import other key dependencies
try:
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    import pandas as pd
    from PIL import Image
    print("All key dependencies imported successfully")
except Exception as e:
    print(f"Dependency import error: {str(e)}")
    st.error(f"Dependency import error: {str(e)}")

# Main application imports
print("Importing main application...")
try:
    # Import our modules
    from utils.preprocessing import preprocess_image, extract_iris_region
    from models.cnn_model import load_cnn_model, extract_features, predict_class
    from models.gabor_model import extract_features as gabor_features
    from models.wavelet_model import extract_features as wavelet_features
    
    # Constants
    MODEL_PATH = "iris_cnn_model.h5"
    CLASS_MAPPING_FILE = "class_mapping.json"
    UPLOAD_DIR = "uploads"
    
    # Make sure upload directory exists
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    print("Importing original app.py...")
    # Import the main app content
    from app import apply_custom_css, sidebar_menu, show_home, show_recognition, show_about
    
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
        elif st.session_state.menu == "Recognition":
            show_recognition()
        elif st.session_state.menu == "About":
            show_about()

    # Run the app
    if __name__ == "__main__":
        main()
        
except Exception as e:
    print(f"Application import error: {str(e)}")
    st.title("Iris Recognition System")
    st.error(f"Error loading application: {str(e)}")
    st.info("This may be due to compatibility issues with the current Python environment. Please check the logs for more information.")
