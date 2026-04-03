import tensorflow as tf
from tensorflow.keras import layers, models, applications

def create_model(model_type='custom', num_classes=38, input_shape=(224, 224, 3)):
    """
    Creates either a custom CNN or a transfer learning model.
    """
    # 1. Data Augmentation block to prevent overfitting
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical", input_shape=input_shape),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2),
    ], name='data_augmentation')

    if model_type == 'custom':
        model = models.Sequential([
            data_augmentation,
            
            # Convolutional Block 1
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D(2, 2),
            
            # Convolutional Block 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D(2, 2),
            
            # Convolutional Block 3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D(2, 2),
            
            # Convolutional Block 4
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D(2, 2),
            
            # Flatten and Fully Connected Layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
    
    elif model_type == 'transfer':
        # Transfer Learning using MobileNetV2
        base_model = applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze the base model to prevent destroying pre-trained weights
        base_model.trainable = False
        
        model = models.Sequential([
            data_augmentation,
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        
    else:
        raise ValueError("model_type must be either 'custom' or 'transfer'")

    return model
