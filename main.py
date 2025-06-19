import os
import sys
import streamlit as st

# Add the project root directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
    print(f"Added {current_dir} to Python path")

# Display a brief loading message
st.set_page_config(
    page_title="Iris Recognition System",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Setup progress bar
progress_bar = st.progress(0)
st.markdown("# Iris Recognition System")
st.markdown("### Loading application...")

# Check imports step by step
try:
    progress_bar.progress(10)
    st.write("Importing common libraries...")
    import numpy as np
    import tensorflow as tf
    import cv2
    import matplotlib.pyplot as plt
    import pandas as pd
    from PIL import Image
    progress_bar.progress(30)
    
    st.write("Importing utils module...")
    # Import utils module
    try:
        import utils
        from utils.preprocessing import preprocess_image, extract_iris_region
        st.success("✅ Utils module loaded successfully")
        progress_bar.progress(50)
    except ImportError as e:
        st.error(f"❌ Error importing utils module: {e}")
        # Try alternative import approach
        st.write("Trying alternative import approach...")
        try:
            # Handle PYTHONPATH manually
            utils_dir = os.path.join(current_dir, "utils")
            if os.path.exists(utils_dir):
                if utils_dir not in sys.path:
                    sys.path.insert(0, utils_dir)
                    st.write(f"Added {utils_dir} to Python path")
                # Try direct import from the preprocessing file
                import preprocessing
                from preprocessing import preprocess_image, extract_iris_region
                st.success("✅ Utils imported using alternative approach")
            else:
                st.error(f"❌ Utils directory not found at {utils_dir}")
                st.write("Directories in root:")
                st.write(os.listdir(current_dir))
                raise ImportError(f"Utils directory not found at {utils_dir}")
        except ImportError as e2:
            st.error(f"❌ Alternative import approach failed: {e2}")
            raise e2
    
    progress_bar.progress(70)
    st.write("Importing models module...")
    # Import models module
    try:
        import models
        from models.cnn_model import load_cnn_model, extract_features, predict_class
        from models.gabor_model import extract_features as gabor_features
        from models.wavelet_model import extract_features as wavelet_features
        st.success("✅ Models module loaded successfully")
    except ImportError as e:
        st.error(f"❌ Error importing models module: {e}")
        # Try alternative import approach
        st.write("Trying alternative import approach...")
        try:
            # Handle PYTHONPATH manually
            models_dir = os.path.join(current_dir, "models")
            if os.path.exists(models_dir):
                if models_dir not in sys.path:
                    sys.path.insert(0, models_dir)
                    st.write(f"Added {models_dir} to Python path")
                # Try direct imports from model files
                import cnn_model
                from cnn_model import load_cnn_model, extract_features, predict_class
                import gabor_model
                from gabor_model import extract_features as gabor_features
                import wavelet_model
                from wavelet_model import extract_features as wavelet_features
                st.success("✅ Models imported using alternative approach")
            else:
                st.error(f"❌ Models directory not found at {models_dir}")
                st.write("Directories in root:")
                st.write(os.listdir(current_dir))
                raise ImportError(f"Models directory not found at {models_dir}")
        except ImportError as e2:
            st.error(f"❌ Alternative import approach failed: {e2}")
            raise e2
    
    progress_bar.progress(90)
    st.write("Starting application...")
    
    # All modules loaded successfully, import and run the main app
    import app
    progress_bar.progress(100)
    
    # Clear the loading elements
    progress_bar.empty()
    st.empty()
    
    # Run the app
    if hasattr(app, 'main'):
        app.main()
    else:
        st.error("Main function not found in app.py")
        
except Exception as e:
    st.error(f"❌ Error: {str(e)}")
    st.write("### Debug Information")
    st.write(f"Python version: {sys.version}")
    st.write(f"Python path: {sys.path}")
    st.write(f"Current directory: {os.getcwd()}")
    st.write(f"Directory contents: {os.listdir('.')}")
    
    if os.path.exists("utils"):
        st.write(f"Utils directory exists: {os.listdir('utils')}")
    else:
        st.error("Utils directory not found!")
        
    if os.path.exists("models"):
        st.write(f"Models directory exists: {os.listdir('models')}")
    else:
        st.error("Models directory not found!")
        
    # Show the app.py content if it exists
    if os.path.exists("app.py"):
        try:
            with open("app.py", "r") as f:
                app_content = f.read()
            st.write("### app.py content:")
            st.code(app_content[:1000] + "..." if len(app_content) > 1000 else app_content)
        except Exception as read_error:
            st.error(f"Could not read app.py: {str(read_error)}")
