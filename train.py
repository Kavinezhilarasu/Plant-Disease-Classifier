import os
import tensorflow as tf
import matplotlib.pyplot as plt
from utils.data_loader import load_data
from utils.model_architectures import create_model

# Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10  # Adjust as needed
USE_SUBSET = True  # True for quick local testing, False for full training
MODEL_TYPE = 'transfer'  # Options: 'custom' or 'transfer'

def plot_history(history):
    """Saves training history graphs."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plt.savefig('models/training_history.png')
    print("Saved training history plot to models/training_history.png")

def main():
    print(f"Starting training pipeline...")
    print(f"Model Type: {MODEL_TYPE}")
    print(f"Using Subset: {USE_SUBSET}")
    
    # 1. Load Data
    train_ds, val_ds, test_ds, info = load_data(
        img_size=IMG_SIZE, 
        batch_size=BATCH_SIZE, 
        use_subset=USE_SUBSET
    )
    
    num_classes = info.features['label'].num_classes
    
    # 2. Compile Model
    model = create_model(model_type=MODEL_TYPE, num_classes=num_classes, input_shape=IMG_SIZE + (3,))
    
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    model.summary()
    
    # 3. Callbacks
    # Save the best model during training
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='models/plant_disease_model.keras',
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )
    
    # Stop earlier if the model stops improving
    early_stop_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
    
    # 4. Train
    print("Beginning model training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[checkpoint_callback, early_stop_callback]
    )
    
    # 5. Evaluate and Visualize
    plot_history(history)
    print("Training finished successfully!")

if __name__ == '__main__':
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    main()
