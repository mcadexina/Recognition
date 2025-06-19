import streamlit as st
import os
import importlib
import sys

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
    print(f"Added {current_dir} to Python path")

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

# Show details about environment
st.title("Iris Recognition System")
st.write(f"Python version: {sys.version}")
st.write(f"Current directory: {current_dir}")
st.write("Checking module availability...")

# Check if modules are available
module_paths = {
    "utils": os.path.join(current_dir, "utils"),
    "models": os.path.join(current_dir, "models")
}

for module_name, module_path in module_paths.items():
    if os.path.exists(module_path):
        st.success(f"✅ Module directory '{module_name}' found at {module_path}")
    else:
        st.error(f"❌ Module directory '{module_name}' not found at {module_path}")

# List files in current directory for debugging
st.write("Files in current directory:")
st.code('\n'.join(os.listdir(current_dir)))

# Main application imports
print("Importing main application...")
try:    # Import our modules
    print("Checking utils path:", os.path.join(current_dir, "utils"))
    print("Files in directory:", os.listdir(current_dir))
    
    # Check if utils/models directories exist
    if not os.path.exists(os.path.join(current_dir, "utils")) or not os.path.exists(os.path.join(current_dir, "models")):
        st.warning("Utils or models directories not found! Switching to fallback mode.")
        raise ImportError("Required module directories not found")
        
    # Import utility functions directly
    sys.path.insert(0, os.path.join(current_dir, "utils"))
    sys.path.insert(0, os.path.join(current_dir, "models"))
    
    # Try direct imports
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
    st.warning(f"Error loading main application: {str(e)}")
    st.info("Loading fallback application...")
    
    try:
        # Try to load the fallback app instead
        if os.path.exists(os.path.join(current_dir, "fallback_app.py")):
            import fallback_app
            fallback_app.main()
        else:
            st.error("Fallback application not found!")
            st.info("This may be due to compatibility issues with the current Python environment. Please check the logs for more information.")
    except Exception as fallback_error:
        st.error(f"Error loading fallback application: {str(fallback_error)}")
        st.info("Please contact support for assistance.")
