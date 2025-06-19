import streamlit as st
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from PIL import Image
import io
import base64
import json

# Set page config
st.set_page_config(
    page_title="Iris Recognition System",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Display environment info
st.sidebar.write("System Info:")
st.sidebar.code(f"Python: {sys.version.split()[0]}")

try:
    import tensorflow as tf
    st.sidebar.success(f"TensorFlow: {tf.__version__}")
except ImportError:
    st.sidebar.error("TensorFlow not available")

try:
    import cv2
    st.sidebar.success(f"OpenCV: {cv2.__version__}")
except ImportError:
    st.sidebar.error("OpenCV not available")

# Create uploads directory
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

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
    menu = st.sidebar.radio("Go to", ["Home", "Recognition Demo", "About"])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("## System Status")
    
    # Display current directory structure
    with st.sidebar.expander("Directory Structure"):
        try:
            dirs = [d for d in os.listdir('.') if os.path.isdir(d)]
            st.code("\n".join(dirs))
        except:
            st.error("Could not list directories")
    
    return menu

# Define simplified image processing functions
def preprocess_image(image):
    """Simple preprocessing for demo purposes"""
    if isinstance(image, Image.Image):
        # Convert PIL to numpy array
        img = np.array(image)
    else:
        img = image
        
    # Convert to grayscale if needed
    if len(img.shape) > 2 and img.shape[2] > 1:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
        
    # Resize for display
    resized = cv2.resize(gray, (320, 240))
    
    return resized

def extract_iris_region(image, demo_mode=True):
    """Extract iris region (simplified for demo)"""
    if demo_mode:
        # In demo mode, just crop the center of the image for visualization
        h, w = image.shape
        center_x, center_y = w // 2, h // 2
        radius = min(w, h) // 4
        
        # Create a circular mask
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        mask = dist_from_center <= radius
        
        masked_img = image.copy()
        masked_img[~mask] = 0
        
        # Create a visualization image
        vis_img = np.zeros((h, w, 3), dtype=np.uint8)
        vis_img[:,:,0] = image  # Red channel
        vis_img[:,:,1] = image  # Green channel
        vis_img[:,:,2] = image  # Blue channel
        
        # Draw circle on visualization
        cv2.circle(vis_img, (center_x, center_y), radius, (0, 255, 0), 2)
        
        return masked_img, vis_img
    else:
        # Placeholder for actual iris extraction
        return image, image

def show_home():
    st.markdown('<h1 class="main-header">Iris Recognition System</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown('<p class="description">Welcome to the Iris Recognition System. This application demonstrates biometric identification using iris patterns.</p>', unsafe_allow_html=True)
        
        st.markdown('<div class="step-box"><h3>How It Works</h3><ol><li>Upload an eye image</li><li>The system extracts the iris region</li><li>Features are extracted using various techniques</li><li>The system matches against known iris patterns</li></ol></div>', unsafe_allow_html=True)
        
    with col2:
        # Use a sample image from the web
        st.image("https://upload.wikimedia.org/wikipedia/commons/2/27/Iris_-_left_eye_of_a_girl.jpg", caption="Sample Iris Image", width=300)
    
    st.markdown('<div class="warning-box">⚠️ <strong>Demo Mode:</strong> The system is currently running in demo mode with limited functionality.</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### About Iris Recognition
    
    Iris recognition is a method of biometric authentication that uses pattern recognition techniques on images of the iris of an eye. 
    The iris patterns are unique and remain stable throughout a person's life, making them ideal for identification purposes.
    
    ### Features of this System:
    
    - **Image Upload**: Process your own eye images
    - **Iris Extraction**: Isolate the iris region from the eye
    - **Feature Visualization**: See the unique features of each iris
    - **Demo Mode**: Experience the workflow without complex processing
    """)

def show_recognition_demo():
    st.markdown('<h1 class="main-header">Iris Recognition Demo</h1>', unsafe_allow_html=True)
    
    st.markdown('<div class="warning-box">⚠️ This is a simplified demonstration with limited functionality.</div>', unsafe_allow_html=True)
    
    # Image upload
    uploaded_file = st.file_uploader("Upload an eye image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        try:
            # Read image
            image = Image.open(uploaded_file)
            
            # Save original image
            timestamp = int(time.time())
            original_path = os.path.join(UPLOAD_DIR, f"original_{timestamp}.png")
            image.save(original_path)
            
            # Convert to array for processing
            image_np = np.array(image)
            
            # Display original
            st.subheader("Original Image")
            st.image(image, width=300)
            
            # Process stages
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Preprocessing")
                try:
                    # Preprocess
                    processed_img = preprocess_image(image_np)
                    st.image(processed_img, width=300)
                    
                    # Save processed
                    processed_path = os.path.join(UPLOAD_DIR, f"processed_{timestamp}.png")
                    cv2.imwrite(processed_path, processed_img)
                except Exception as e:
                    st.error(f"Preprocessing error: {str(e)}")
            
            with col2:
                st.subheader("Iris Extraction")
                try:
                    # Extract iris
                    iris_region, vis_img = extract_iris_region(processed_img)
                    st.image(vis_img, width=300)
                    
                    # Save iris
                    iris_path = os.path.join(UPLOAD_DIR, f"iris_{timestamp}.png")
                    cv2.imwrite(iris_path, iris_region)
                except Exception as e:
                    st.error(f"Iris extraction error: {str(e)}")
            
            # Feature visualization (simplified)
            st.subheader("Feature Visualization")
            try:
                # Create a simple feature visualization
                plt.figure(figsize=(10, 4))
                
                # Plot 1: Histogram
                plt.subplot(1, 2, 1)
                plt.hist(iris_region.flatten(), bins=50)
                plt.title("Intensity Histogram")
                
                # Plot 2: Sample "features"
                plt.subplot(1, 2, 2)
                
                # Generate some random "features" for demo
                n_features = 16
                feature_values = np.random.random(n_features) * 0.5 + 0.5
                plt.bar(range(n_features), feature_values)
                plt.title("Sample Features")
                plt.ylim(0, 1)
                
                # Save figure
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                
                # Display figure
                st.pyplot(plt)
                
                # Save features image
                features_path = os.path.join(UPLOAD_DIR, f"features_{timestamp}.png")
                with open(features_path, 'wb') as f:
                    f.write(buf.getvalue())
                
            except Exception as e:
                st.error(f"Feature visualization error: {str(e)}")
            
            # Demo results
            st.subheader("Recognition Results")
            st.markdown('<div class="success-box"><strong>Demo Result:</strong> This is a simulated result for demonstration purposes.</div>', unsafe_allow_html=True)
            
            # Show a simple confidence score
            confidence = np.random.random() * 30 + 70  # Random score between 70-100
            st.metric("Confidence Score", f"{confidence:.1f}%")
            
            # Display randomly selected simulated class
            classes = ["Person 1", "Person 2", "Person 3", "Unknown"]
            selected_class = classes[np.random.randint(0, len(classes))]
            
            if selected_class == "Unknown":
                st.warning(f"Result: {selected_class}")
            else:
                st.success(f"Result: {selected_class}")
                
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.info("Please try uploading a different image.")
    else:
        # Show sample image and instructions
        st.info("Please upload an eye image to begin the recognition process.")
        st.markdown("#### Sample Results")
        
        # Display a sample result
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.image("https://upload.wikimedia.org/wikipedia/commons/2/27/Iris_-_left_eye_of_a_girl.jpg", width=200, caption="Original Eye Image")
        
        with col2:
            # Create a sample processed image
            sample_img = np.zeros((200, 200), dtype=np.uint8)
            cv2.circle(sample_img, (100, 100), 80, 255, 1)
            cv2.circle(sample_img, (100, 100), 50, 255, 1)
            st.image(sample_img, width=200, caption="Processed Iris")
        
        with col3:
            # Create a sample feature visualization
            plt.figure(figsize=(3, 3))
            plt.bar(range(8), np.random.random(8) * 0.5 + 0.5)
            plt.title("Sample Features")
            plt.ylim(0, 1)
            st.pyplot(plt)

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
    
    ### GitHub Repository
    
    This project is available on GitHub at: [https://github.com/mcadexina/Recognition](https://github.com/mcadexina/Recognition)
    """)
    
    st.markdown("### Current Environment")
    st.code(f"""
    Python version: {sys.version}
    Current working directory: {os.getcwd()}
    Files in directory: {', '.join(os.listdir('.')[:10])}{'...' if len(os.listdir('.')) > 10 else ''}
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
    elif st.session_state.menu == "Recognition Demo":
        show_recognition_demo()
    elif st.session_state.menu == "About":
        show_about()

# Run the app
if __name__ == "__main__":
    main()
