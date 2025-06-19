# utils/preprocessing.py
import cv2
import numpy as np
from scipy import ndimage

def preprocess_image(img, target_size=(64, 64), enhance=True, keep_color=False):
    """
    Preprocess an iris image for CNN model
    
    Args:
        img: Input image (NumPy array or file path)
        target_size: Target size for resizing (default: 64x64)
        enhance: Whether to apply enhancement techniques
        keep_color: Whether to keep the color channels (default: False - convert to grayscale)
        
    Returns:
        Preprocessed image as numpy array
    """
    # Handle loading image from file if needed
    if isinstance(img, str):
        img = cv2.imread(img)
        if img is None:
            raise ValueError(f"Failed to load image from {img}")
    
    # Input validation
    if img is None:
        raise ValueError("Input image is None")
    if not isinstance(img, np.ndarray):
        raise TypeError("Expected a NumPy array")
    if img.ndim not in [2, 3]:
        raise ValueError(f"Unexpected image dimensions: {img.ndim}")
    
    # Make a copy to avoid modifying the original
    original_img = img.copy()
    
    # Convert BGR to RGB if it's a color image (OpenCV loads as BGR)
    if img.ndim == 3 and img.shape[2] >= 3:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        rgb_img = img
        
    # Convert to grayscale if needed and not keeping color
    if not keep_color:
        if img.ndim == 3:
            # Save a copy of the grayscale image for processing
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
            img = gray_img
        else:
            gray_img = img
    else:
        # If keeping color, ensure we have 3 channels
        if img.ndim == 2:
            # Convert grayscale to RGB
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 1:  # Single channel image with shape (H, W, 1)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        # Note: img remains as-is if it's already a 3-channel image
      # Apply enhancement techniques
    if enhance:
        if not keep_color:  # Grayscale enhancement
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img = clahe.apply(img)
            
            # Apply Gaussian blur for noise reduction
            img = cv2.GaussianBlur(img, (5, 5), 0)
            
            # Image sharpening using unsharp masking
            gaussian_3 = cv2.GaussianBlur(img, (9, 9), 10.0)
            img = cv2.addWeighted(img, 1.5, gaussian_3, -0.5, 0)
        else:  # Color enhancement
            # Convert to LAB color space for CLAHE
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge back the channels
            lab = cv2.merge((l, a, b))
            
            # Convert back to RGB
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            # Apply light Gaussian blur for noise reduction
            img = cv2.GaussianBlur(img, (3, 3), 0)
            
            # Apply sharpening
            kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])
            img = cv2.filter2D(img, -1, kernel)
    
    # Resize to target size
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    
    # Normalize to [0, 1]
    img = img / 255.0
    
    # Add channel dimension for grayscale if needed
    if not keep_color and len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)
    
    return img

def extract_iris_region(img):
    """
    Extract the iris region using Hough Circle Transform
    
    Args:
        img: Input eye image
        
    Returns:
        Extracted iris region or None if iris not found
    """
    if isinstance(img, str):
        img = cv2.imread(img)
        if img is None:
            raise ValueError(f"Failed to load image from {img}")
    
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Apply blur and equalization for better circle detection
    gray = cv2.medianBlur(gray, 5)
    gray = cv2.equalizeHist(gray)
    
    # Apply Hough Circles to detect iris
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=20,
        maxRadius=100
    )
    
    if circles is None:
        return None
    
    # Take the most prominent circle (presumably the iris)
    circles = np.uint16(np.around(circles))
    circle = circles[0, 0]  # Get the first circle
    
    # Extract the iris region
    center_x, center_y, radius = circle
    
    # Create a mask for the iris
    mask = np.zeros_like(gray)
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)
    
    # Apply the mask to extract the iris
    iris_region = cv2.bitwise_and(gray, mask)
    
    # Crop to a square around the circle
    x1 = max(0, center_x - radius)
    y1 = max(0, center_y - radius)
    x2 = min(img.shape[1], center_x + radius)
    y2 = min(img.shape[0], center_y + radius)
    
    cropped_iris = iris_region[y1:y2, x1:x2]
    
    return cropped_iris

def data_augmentation(img):
    """
    Apply random augmentation to an image for training
    
    Args:
        img: Input image array
        
    Returns:
        Augmented image
    """
    # Random rotation (slight)
    angle = np.random.uniform(-10, 10)
    img = ndimage.rotate(img, angle, reshape=False)
    
    # Random brightness/contrast adjustment
    alpha = np.random.uniform(0.7, 1.3)  # contrast
    beta = np.random.uniform(-0.1, 0.1)  # brightness
    img = np.clip(alpha * img + beta, 0, 1)
    
    # Random noise
    if np.random.random() > 0.5:
        noise = np.random.normal(0, 0.01, img.shape)
        img = np.clip(img + noise, 0, 1)
    
    return img

