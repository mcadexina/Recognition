import tensorflow as tf
from keras import layers, models
import numpy as np

# Create a CNN model for iris image classification or feature extraction
# Can handle both grayscale (1 channel) and RGB (3 channels) images
def create_cnn_model(input_shape=(64, 64, 1), num_classes=10):
    # Check if the input_shape has the channel dimension
    if len(input_shape) == 2:
        # Add a channel dimension for grayscale
        input_shape = (*input_shape, 1)
    
    # Check number of channels (1 for grayscale, 3 for RGB)
    if input_shape[-1] not in [1, 3]:
        print(f"Warning: Unexpected number of channels: {input_shape[-1]}. " +
              "Expected 1 (grayscale) or 3 (RGB). Defaulting to 1 channel.")
        input_shape = (*input_shape[:2], 1)
    
    print(f"Creating CNN model with input shape: {input_shape}")
    
    model = models.Sequential([
        # First convolutional block with batch normalization
        layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Fully connected layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu', name="feature_layer"),  # Feature extraction layer
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])    # Use a learning rate schedule for better convergence
    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=10000,
        decay_rate=0.9,
        staircase=True)
        
    # Compile with better metrics
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    
    return model

# Load model from file or create a new one
def load_cnn_model(model_path=None, input_shape=(64, 64, 1), num_classes=10, debug=True):
    if model_path:
        try:
            model = models.load_model(model_path)
            
            if debug:
                print(f"\nLoaded model summary:")
                try:
                    model.summary()
                except Exception as summary_error:
                    print(f"Could not print model summary: {summary_error}")
                
                # Safely get model input shape
                try:
                    print(f"\nModel input shape: {model.input_shape}")
                except Exception as input_shape_error:
                    print(f"Could not get model input shape: {input_shape_error}")
                
                # More safely get layer information
                print("\nLayer shapes:")
                for i, layer in enumerate(model.layers):
                    try:
                        # For TensorFlow 2.x layers, we need to be very careful here
                        layer_type = type(layer).__name__
                        layer_name = layer.name if hasattr(layer, 'name') else "unknown"
                        
                        # Get input shape - different approaches
                        input_shape = None
                        if hasattr(layer, '_batch_input_shape') and layer._batch_input_shape is not None:
                            input_shape = str(layer._batch_input_shape)
                        elif hasattr(layer, 'input_spec') and layer.input_spec is not None:
                            if isinstance(layer.input_spec, list) and len(layer.input_spec) > 0:
                                if hasattr(layer.input_spec[0], 'shape') and layer.input_spec[0].shape is not None:
                                    input_shape = str(layer.input_spec[0].shape)
                        
                        # Get output shape
                        output_shape = str(layer.output_shape) if hasattr(layer, 'output_shape') else "Not available"
                        
                        print(f"Layer {i}: {layer_name}, Type: {layer_type}")
                        print(f"  Input: {input_shape if input_shape else 'Not available'}")
                        print(f"  Output: {output_shape}")
                    except Exception as layer_error:
                        print(f"  Error getting info for layer {i}: {layer_error}")
            
            return model
        except Exception as e:
            print(f"Model loading error details: {str(e)}")
            import traceback
            print(traceback.format_exc())
            print("Creating a new model instead...")
            return create_cnn_model(input_shape, num_classes)
    
    print(f"Creating new model with input_shape={input_shape}, num_classes={num_classes}")
    return create_cnn_model(input_shape, num_classes)

# Preprocess an image for prediction - handles both grayscale and RGB
def preprocess_image(image, target_size=(64, 64), force_channels=None):
    """
    Preprocess an image for model prediction, ensuring proper tensor formatting.
    
    Args:
        image: Input image as numpy array
        target_size: Target size for resizing (default: 64x64)
        force_channels: Force specific number of channels (1=grayscale, 3=RGB)
                        If None, keeps original channels with proper formatting
    
    Returns:
        TensorFlow tensor ready for model input (batch dimension included)
    """
    print(f"CNN preprocess_image: Input image shape: {image.shape if hasattr(image, 'shape') else 'unknown'}")
    
    # Convert to numpy array if not already
    if not isinstance(image, np.ndarray):
        try:
            image = np.array(image)
        except Exception as e:
            print(f"Error converting image to numpy array: {e}")
            # Create a dummy array as fallback
            image = np.zeros((*target_size, 1), dtype=np.float32)
    
    # Convert to tensor first (preserves original format)
    tensor_img = tf.convert_to_tensor(image)
    
    # Handle different image dimensions
    if len(tensor_img.shape) == 2:  # Single grayscale image
        print("Processing 2D grayscale image")
        if force_channels == 3:
            # Convert grayscale to RGB
            tensor_img = tf.image.grayscale_to_rgb(tensor_img[..., tf.newaxis])
        else:
            # Add channel dimension for grayscale
            tensor_img = tensor_img[..., tf.newaxis]
    
    elif len(tensor_img.shape) == 3:
        if tensor_img.shape[-1] == 1:  # Already has channel dimension
            print("Processing 3D grayscale image (with channel dimension)")
            if force_channels == 3:
                # Convert single channel to RGB
                tensor_img = tf.image.grayscale_to_rgb(tensor_img)
        
        elif tensor_img.shape[-1] == 3:  # RGB image
            print("Processing RGB image")
            if force_channels == 1:
                # Convert RGB to grayscale
                tensor_img = tf.image.rgb_to_grayscale(tensor_img)
        
        elif tensor_img.shape[-1] == 4:  # RGBA image
            print("Processing RGBA image - removing alpha channel")
            # Remove alpha channel
            tensor_img = tensor_img[..., :3]
            if force_channels == 1:
                tensor_img = tf.image.rgb_to_grayscale(tensor_img)
        
        else:
            print(f"Unusual channel count: {tensor_img.shape[-1]}. Creating fallback tensor.")
            # Create a dummy tensor with the right dimensions as fallback
            tensor_img = tf.zeros((*target_size, 1 if force_channels == 1 else 3), dtype=tf.float32)
    
    else:
        print(f"Unsupported tensor shape: {tensor_img.shape}. Creating fallback tensor.")
        # Create a dummy tensor with the right dimensions as fallback
        tensor_img = tf.zeros((*target_size, 1 if force_channels == 1 else 3), dtype=tf.float32)
    
    # Resize to target size
    try:
        tensor_img = tf.image.resize(tensor_img, target_size)
    except Exception as resize_error:
        print(f"Error resizing image: {resize_error}")
        # Create a properly sized tensor as fallback
        tensor_img = tf.zeros((*target_size, tensor_img.shape[-1]), dtype=tf.float32)
    
    # Normalize to [0, 1]
    tensor_img = tf.cast(tensor_img, tf.float32) / 255.0
    
    # Add batch dimension
    tensor_img = tf.expand_dims(tensor_img, axis=0)
    
    print(f"Final preprocessed tensor shape: {tensor_img.shape}")
    return tensor_img

# Predict class index using the trained CNN
def predict_class(image, model):
    try:
        # Safely determine the input shape and channels expected by the model
        target_size = (64, 64)  # Default size
        expected_channels = None  # Will determine from model
        
        try:
            # Try to get input shape from model if available
            if hasattr(model, 'input_shape') and model.input_shape is not None:
                if len(model.input_shape) > 1:
                    # Get target size from model shape
                    target_size = (model.input_shape[1], model.input_shape[2])
                    # Get expected channels if available
                    if len(model.input_shape) > 3 and model.input_shape[3] is not None:
                        expected_channels = model.input_shape[3]
                        print(f"Model expects {expected_channels} channel(s) from input_shape")
            
            # Alternatively, try to get from the first layer
            elif hasattr(model.layers[0], '_batch_input_shape') and model.layers[0]._batch_input_shape is not None:
                if len(model.layers[0]._batch_input_shape) > 1:
                    # Get target size from first layer
                    target_size = (model.layers[0]._batch_input_shape[1], model.layers[0]._batch_input_shape[2])
                    # Get expected channels if available
                    if len(model.layers[0]._batch_input_shape) > 3 and model.layers[0]._batch_input_shape[3] is not None:
                        expected_channels = model.layers[0]._batch_input_shape[3]
                        print(f"Model expects {expected_channels} channel(s) from first layer")
        except Exception as shape_error:
            print(f"Error determining input shape: {shape_error}")
            print("Using default target size (64, 64) and inferred channels")
        
        print(f"Using target size for prediction: {target_size}")
        
        # Check image type and adjust channels if needed
        if expected_channels:
            print(f"Preprocessing image with forced channels: {expected_channels}")
            # Preprocess with forced channels to match model expectation
            processed_image = preprocess_image(image, target_size=target_size, force_channels=expected_channels)
        else:
            # Let preprocess_image handle automatic channel determination
            processed_image = preprocess_image(image, target_size=target_size)
        
        # Get input shape after preprocessing
        print(f"Processed image shape: {processed_image.shape}")
        
        # Make prediction with proper error handling
        try:
            prediction = model.predict(processed_image, verbose=0)
            predicted_class = tf.argmax(prediction, axis=1).numpy()[0]
            return predicted_class
        except Exception as predict_error:
            print(f"Error in model prediction: {predict_error}")
            print("Attempting alternate prediction approach...")
            
            # Try a different preprocessing approach as fallback
            try:
                # Try with opposite channel setting
                if expected_channels == 1:
                    processed_image = preprocess_image(image, target_size=target_size, force_channels=3)
                else:
                    processed_image = preprocess_image(image, target_size=target_size, force_channels=1)
                
                print(f"Fallback processed image shape: {processed_image.shape}")
                prediction = model.predict(processed_image, verbose=0)
                predicted_class = tf.argmax(prediction, axis=1).numpy()[0]
                return predicted_class
            except:
                print("Fallback prediction also failed. Returning default class 0.")
                return 0  # Return default class as last resort
        
    except Exception as e:
        print(f"Error in predict_class: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return 0  # Return default class 0 in case of error

# Build a feature extractor from the model by removing the softmax layer
def build_feature_extractor(model):
    try:
        feature_layer = None
        feature_layer_name = None
        
        # Try to find a layer with "feature" in the name first
        for layer in model.layers:
            if hasattr(layer, 'name') and "feature" in layer.name.lower():
                feature_layer = layer
                feature_layer_name = layer.name
                break
        
        # If no feature layer found by name, use the second-to-last dense layer
        if feature_layer is None:
            # Find all dense layers
            dense_layers = []
            dense_layer_names = []
            
            for i, layer in enumerate(model.layers):
                if isinstance(layer, tf.keras.layers.Dense):
                    dense_layers.append(layer)
                    dense_layer_names.append(layer.name)
            
            print(f"Found {len(dense_layers)} dense layers: {dense_layer_names}")
            
            # If we have at least 2 dense layers, use the second-to-last one
            if len(dense_layers) >= 2:
                feature_layer = dense_layers[-2]  # Second-to-last dense layer
                feature_layer_name = feature_layer.name
            # If we only have 1 dense layer, use it
            elif len(dense_layers) == 1:
                feature_layer = dense_layers[0]
                feature_layer_name = feature_layer.name
            # If no dense layers found, use the last non-output layer
            elif len(model.layers) >= 2:
                # Get the layer before the output layer as a fallback
                feature_layer = model.layers[-2]
                feature_layer_name = feature_layer.name
        
        if feature_layer is None:
            raise ValueError("No suitable feature extraction layer found in model")
        
        print(f"Using layer '{feature_layer_name}' for feature extraction")
        
        # Create feature extraction model by name rather than by object
        # This approach works more reliably across TF versions
        try:
            # Find the layer index by name
            layer_idx = None
            for i, layer in enumerate(model.layers):
                if layer.name == feature_layer_name:
                    layer_idx = i
                    break
                    
            if layer_idx is None:
                raise ValueError(f"Could not find layer with name {feature_layer_name}")
                
            # Create a new model that outputs the feature layer
            feature_model = tf.keras.models.Model(
                inputs=model.input,
                outputs=model.layers[layer_idx].output
            )
            return feature_model
            
        except Exception as e:
            print(f"Error creating feature model by index: {e}")
            # Third approach - last resort
            try:
                # Create model using get_layer method
                feature_layer = model.get_layer(feature_layer_name)
                feature_model = tf.keras.models.Model(inputs=model.input, outputs=feature_layer.output)
                return feature_model
            except:
                # If all else fails, use second-to-last layer
                if len(model.layers) >= 2:
                    feature_model = tf.keras.models.Model(
                        inputs=model.input,
                        outputs=model.layers[-2].output
                    )
                    print(f"Using fallback layer {model.layers[-2].name} for feature extraction")
                    return feature_model
                else:
                    raise ValueError("Could not create feature extractor model")
        
    except Exception as e:
        print(f"Error in build_feature_extractor: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise ValueError(f"Could not build feature extractor: {str(e)}")


# Extract deep features from an image using the CNN feature extractor
def extract_features(image, model):
    try:
        # Safely determine the input shape and channels expected by the model
        target_size = (64, 64)  # Default size
        expected_channels = None  # Will determine from model
        
        try:
            # Try to get input shape from model if available
            if hasattr(model, 'input_shape') and model.input_shape is not None:
                if len(model.input_shape) > 1:
                    # Get target size from model shape
                    target_size = (model.input_shape[1], model.input_shape[2])
                    # Get expected channels if available
                    if len(model.input_shape) > 3 and model.input_shape[3] is not None:
                        expected_channels = model.input_shape[3]
                        print(f"Model expects {expected_channels} channel(s) from input_shape")
            
            # Alternatively, try to get from the first layer
            elif hasattr(model.layers[0], '_batch_input_shape') and model.layers[0]._batch_input_shape is not None:
                if len(model.layers[0]._batch_input_shape) > 1:
                    # Get target size from first layer
                    target_size = (model.layers[0]._batch_input_shape[1], model.layers[0]._batch_input_shape[2])
                    # Get expected channels if available
                    if len(model.layers[0]._batch_input_shape) > 3 and model.layers[0]._batch_input_shape[3] is not None:
                        expected_channels = model.layers[0]._batch_input_shape[3]
                        print(f"Model expects {expected_channels} channel(s) from first layer")
        except Exception as shape_error:
            print(f"Error determining input shape: {shape_error}")
            print("Using default target size (64, 64) and inferred channels")
        
        print(f"Using target size for feature extraction: {target_size}")
        
        # Check image type and adjust channels if needed
        if expected_channels:
            print(f"Preprocessing image with forced channels: {expected_channels}")
            # Preprocess with forced channels to match model expectation
            processed_image = preprocess_image(image, target_size=target_size, force_channels=expected_channels)
        else:
            # Let preprocess_image handle automatic channel determination
            processed_image = preprocess_image(image, target_size=target_size)
        
        # Print shape for debugging
        print(f"Processed image shape for feature extraction: {processed_image.shape}")
        
        try:
            # Try to build the feature extractor
            feature_model = build_feature_extractor(model)
            
            # Extract features 
            features = feature_model.predict(processed_image, verbose=0)
            return features.flatten()
        except ValueError as ve:
            # If feature layer not found, try a different approach
            print(f"Feature extraction error: {ve}")
            print("Attempting alternate feature extraction method...")
            
            # Use the output of the layer before the final dense layer as features
            dense_layers = [i for i, layer in enumerate(model.layers) 
                          if isinstance(layer, tf.keras.layers.Dense)]
            
            if len(dense_layers) >= 2:
                # Use the second-to-last dense layer
                feature_layer_idx = dense_layers[-2]
                feature_model = tf.keras.models.Model(
                    inputs=model.input, 
                    outputs=model.layers[feature_layer_idx].output
                )
                print(f"Using layer {model.layers[feature_layer_idx].name} for features")
                features = feature_model.predict(processed_image, verbose=0)
                return features.flatten()
            else:
                # Use the flattened output before any dense layer
                flatten_layers = [i for i, layer in enumerate(model.layers) 
                                if isinstance(layer, tf.keras.layers.Flatten)]
                                
                if flatten_layers:
                    feature_layer_idx = flatten_layers[0]
                    feature_model = tf.keras.models.Model(
                        inputs=model.input, 
                        outputs=model.layers[feature_layer_idx].output
                    )
                    print(f"Using layer {model.layers[feature_layer_idx].name} for features")
                    features = feature_model.predict(processed_image, verbose=0)
                    return features.flatten()
                else:
                    # Last resort: use last conv layer output
                    conv_layers = [i for i, layer in enumerate(model.layers) 
                                 if 'conv' in layer.name.lower()]
                    if conv_layers:
                        feature_layer_idx = conv_layers[-1]
                        feature_model = tf.keras.models.Model(
                            inputs=model.input, 
                            outputs=model.layers[feature_layer_idx].output
                        )
                        print(f"Using layer {model.layers[feature_layer_idx].name} for features")
                        features = feature_model.predict(processed_image, verbose=0)
                        return features.flatten()
                    
            # If all else fails, return zeros
            print("Could not find suitable feature extraction layer")
            return np.zeros(128)
    except Exception as e:
        print(f"Error during feature extraction: {str(e)}")
        import traceback
        print(traceback.format_exc())
        # Return a zero vector as fallback (adjust size as needed)
        return np.zeros(128)
