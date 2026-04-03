import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from utils.data_loader import load_data

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
# Ensure we use exactly the same subset setting to test on the correct data if running quickly
USE_SUBSET = True

def main():
    print("Loading test dataset...")
    # Load test data only from the split we want
    _, _, test_ds, _ = load_data(img_size=IMG_SIZE, batch_size=BATCH_SIZE, use_subset=USE_SUBSET)
    
    print("Loading saved model...")
    try:
        model = tf.keras.models.load_model('models/plant_disease_model.keras')
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Have you finished training yet? Run `python train.py` first.")
        return

    try:
        with open('models/class_names.json', 'r') as f:
            class_names = json.load(f)
    except Exception as e:
        print("class_names.json not found. Run training first to generate it.")
        return

    # To calculate metrics, we need all true labels and predictions
    print("Predicting on test dataset...")
    y_true = []
    y_pred = []
    
    # We unbatch test_ds to extract true vectors cleanly
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())

    print("\n--- Classification Report ---")
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)
    
    print("Generating Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels', fontsize=16)
    plt.ylabel('True Labels', fontsize=16)
    plt.title('Confusion Matrix', fontsize=20)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('models/confusion_matrix.png')
    print("Saved confusion matrix plot to models/confusion_matrix.png")
    
    # Overall Accuracy calculation (can also be seen in the report)
    loss, acc = model.evaluate(test_ds, verbose=0)
    print(f"\nOverall Test Accuracy: {acc * 100:.2f}%")

if __name__ == '__main__':
    main()
