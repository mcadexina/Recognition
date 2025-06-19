import pywt
import numpy as np
import cv2

def extract_features(image):
    """
    Extract features using 2D Complex Wavelet Transform (2D CWT).

    Args:
        image (numpy.ndarray): Preprocessed iris image.

    Returns:
        numpy.ndarray: Feature vector.
    """
    try:
        # Convert to numpy array if it's not already
        if not isinstance(image, np.ndarray):
            image = np.array(image)
            
        # Make sure image is grayscale
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                print("Converting RGB image to grayscale for Wavelet feature extraction")
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif image.shape[2] == 1:
                image = image[:, :, 0]  # Take first channel
        
        # Verify image is 2D at this point
        if len(image.shape) != 2:
            raise ValueError(f"Expected 2D image after preprocessing, got shape: {image.shape}")
            
        print(f"Wavelet feature extraction: working with image shape {image.shape}")

        # Resize image for consistency
        image = cv2.resize(image, (128, 128))

        # Apply 2D Wavelet Transform
        coeffs = pywt.dwt2(image, 'haar')  # You can choose other wavelets like 'db2', 'sym3' if you want
        cA, (cH, cV, cD) = coeffs

        # Flatten and concatenate features
        features = np.hstack((cA.flatten(), cH.flatten(), cV.flatten(), cD.flatten()))
        
        # For additional features, calculate statistics from wavelet coefficients
        additional_features = []
        for coeff in [cA, cH, cV, cD]:
            mean = np.mean(coeff)
            std = np.std(coeff)
            energy = np.sum(coeff**2) / coeff.size
            entropy = -np.sum(coeff**2 * np.log2(coeff**2 + 1e-10)) / coeff.size
            additional_features.extend([mean, std, energy, entropy])
            
        # Combine all features
        all_features = np.hstack((features, additional_features))
        
        return all_features
        
    except Exception as e:
        import traceback
        print(f"Error in Wavelet feature extraction: {str(e)}")
        print(traceback.format_exc())
        # Return a dummy feature vector in case of error
        return np.zeros(200)
