import streamlit as st
import os
import sys

# Display a loading message while we check and import dependencies
st.set_page_config(
    page_title="Iris Recognition System",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="expanded"
)

def check_dependencies():
    """Check for required dependencies and show appropriate messages."""
    try:
        import tensorflow as tf
        st.success(f"✅ TensorFlow {tf.__version__} is available")
        
        try:
            from tensorflow.keras.models import load_model
            st.success("✅ Keras is available")
        except ImportError:
            st.warning("⚠️ tensorflow.keras not found, trying standalone keras")
            try:
                import keras
                from keras.models import load_model
                st.success(f"✅ Standalone Keras {keras.__version__} is available")
            except ImportError:
                st.error("❌ Neither tensorflow.keras nor standalone keras could be imported")
                return False
        
        import numpy as np
        import cv2
        st.success(f"✅ OpenCV {cv2.__version__} is available")
        
        # Check for other key dependencies
        import matplotlib.pyplot as plt
        import pandas as pd
        from PIL import Image
        
        # All dependencies look good
        return True
        
    except ImportError as e:
        st.error(f"❌ Dependency error: {str(e)}")
        return False

# Check if dependencies are available
if check_dependencies():
    st.success("All dependencies loaded successfully. Starting app...")
    # Import the main app
    import app
    # If we get here, import was successful
else:
    st.error("Some dependencies are missing. Please check the error messages above.")
    st.info("You can try to install the missing dependencies using: pip install -r requirements.txt")
