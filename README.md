# Plant Disease Classifier

This project is an end-to-end AI application that uses deep learning to classify plant diseases from images of leaves. It features model training pipelines, comprehensive evaluation scripts, and a modern Streamlit web application for deployment.

## Features
- **Dataset**: Automatically uses the [PlantVillage](https://www.tensorflow.org/datasets/catalog/plant_village) dataset (~54,000 images, 38 classes).
- **Custom CNN Architecture**: A deep Convolutional Neural Network with built-in data augmentation.
- **Transfer Learning Option**: Easy integration with MobileNetV2 for faster convergence and better accuracy.
- **Web App**: A modern, easy-to-use Streamlit web interface for real-time predictions.
- **Evaluation**: Script to generate a Confusion Matrix, and calculate Accuracy, Precision, and Recall.

## Directory Structure
- `app/`: Contains the Streamlit visual UI application (`app.py`).
- `models/`: Saved models and training history plots will be stored here.
- `utils/`: Core utilities for dataset loading (`data_loader.py`) and building architectures (`model_architectures.py`).
- `train.py`: Primary script to train the model.
- `evaluate.py`: Generates the confusion matrix and evaluation metrics.

## Setup Instructions

1. **Install Requirements**
   Run the following command to install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. **Training the Model**
   To train the plant disease classifier, simply run:
   ```bash
   python train.py
   ```
   *Note: By default, the script trains on a 10% subset of the dataset so you can test it locally. To train on the full dataset, modify `USE_SUBSET = False` inside `train.py`.*

3. **Evaluating the Model**
   After training, you can analyze its true performance:
   ```bash
   python evaluate.py
   ```

4. **Running the Web Application**
   Launch the Streamlit app to interact with the trained model:
   ```bash
   streamlit run app/app.py
   ```
