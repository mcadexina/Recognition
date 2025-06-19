# models/gabor_model.py
import numpy as np
import cv2
import math

def log_gabor_filter(shape, omega0=0.5, sigmaF=0.55):
    """Generate Log-Gabor filter in frequency domain."""
    # Handle different shape formats (2D or 3D)
    if len(shape) == 3:
        # For 3D shape (height, width, channels)
        rows, cols = shape[0], shape[1]
    elif len(shape) == 2:
        # For 2D shape (height, width)
        rows, cols = shape
    else:
        raise ValueError(f"Unsupported shape format: {shape}")
    
    x = np.linspace(-0.5, 0.5, cols)
    y = np.linspace(-0.5, 0.5, rows)
    X, Y = np.meshgrid(x, y)
    radius = np.sqrt(X**2 + Y**2)
    radius[int(rows/2), int(cols/2)] = 1  # Avoid log(0)

    log_gabor = np.exp((-(np.log(radius / omega0))**2) / (2 * (np.log(sigmaF))**2))
    log_gabor[int(rows/2), int(cols/2)] = 0  # Center frequency set to zero
    return log_gabor

def extract_features(image):
    """Apply Log-Gabor filter and extract features."""
    try:
        # Convert to numpy array if it's not already
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        # Handle different image formats
        if image.ndim == 3:
            # Convert RGB to grayscale if needed
            if image.shape[2] == 3:
                print("Converting RGB image to grayscale for Gabor feature extraction")
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # Extract the channel if it's a single channel image in 3D format
            elif image.shape[2] == 1:
                image = image[:, :, 0]
        
        # Make sure image is 2D at this point
        if image.ndim != 2:
            raise ValueError(f"Image should be 2D after preprocessing, got shape: {image.shape}")
        
        print(f"Gabor feature extraction: working with image shape {image.shape}")
        
        # Apply FFT
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)

        # Create and apply filter
        gabor = log_gabor_filter(image.shape)
        filtered = np.multiply(f_shift, gabor)

        # Inverse FFT
        filtered_img = np.fft.ifft2(np.fft.ifftshift(filtered))
        magnitude = np.abs(filtered_img)
        
        # Extract statistical features from the filtered image
        # Resize magnitude to standard size for consistent feature vector length
        magnitude_resized = cv2.resize(magnitude, (64, 64))
        
        # Extract features: mean, std, energy, entropy of regions
        features = []
        
        # Divide image into 4x4 blocks and extract features from each
        block_size = 16  # 64/4 = 16
        for i in range(0, 64, block_size):
            for j in range(0, 64, block_size):
                block = magnitude_resized[i:i+block_size, j:j+block_size]
                
                # Calculate statistics
                mean_val = np.mean(block)
                std_val = np.std(block)
                energy = np.sum(block**2) / (block_size**2)
                
                # Calculate entropy
                hist, _ = np.histogram(block, bins=8, range=(0, 1))
                hist = hist / np.sum(hist) if np.sum(hist) > 0 else hist
                hist = hist[hist > 0]  # Remove zeros
                entropy = -np.sum(hist * np.log2(hist)) if len(hist) > 0 else 0
                
                # Add features
                features.extend([mean_val, std_val, energy, entropy])
        
        # Add some global features
        global_mean = np.mean(magnitude_resized)
        global_std = np.std(magnitude_resized)
        global_max = np.max(magnitude_resized)
        global_min = np.min(magnitude_resized)
        
        features.extend([global_mean, global_std, global_max, global_min])
        
        # Return as numpy array
        return np.array(features)
    
    except Exception as e:
        import traceback
        print(f"Error in Gabor feature extraction: {str(e)}")
        print(traceback.format_exc())
        # Return a dummy feature vector in case of error
        return np.zeros(100)

    # Flatten the features
    features = magnitude.flatten()

    return features
