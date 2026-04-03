import tensorflow as tf
import tensorflow_datasets as tfds
import json
import os

def load_data(img_size=(224, 224), batch_size=32, use_subset=False):
    """
    Loads the PlantVillage dataset from tensorflow_datasets.
    Splits into 80% train, 10% validation, 10% test.
    """
    print("Loading PlantVillage dataset...")
    
    # We use roughly 80/10/10 split
    split_config = ['train[:80%]', 'train[80%:90%]', 'train[90%:]']
    
    if use_subset:
        # Use only a tiny fraction (e.g. 1% of total for quick dry runs)
        split_config = ['train[:8%]', 'train[8%:9%]', 'train[9%:10%]']
    
    dataset, info = tfds.load(
        'beans',
        split=split_config,
        with_info=True,
        as_supervised=True
    )
    
    train_ds, val_ds, test_ds = dataset
    
    # Save the class names to a JSON file for the Streamlit app later
    class_names = info.features['label'].names
    with open('models/class_names.json', 'w') as f:
        json.dump(class_names, f)
        
    print(f"Loaded {info.splits['train'].num_examples} total images in full dataset.")
    print(f"Number of classes: {len(class_names)}")
    
    # Preprocessing function
    def preprocess(image, label):
        # Resize image
        image = tf.image.resize(image, img_size)
        # Normalize to [0,1]
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    # Create data pipelines
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.map(preprocess, num_parallel_calls=AUTOTUNE)\
                       .cache()\
                       .shuffle(1000)\
                       .batch(batch_size)\
                       .prefetch(buffer_size=AUTOTUNE)

    val_ds = val_ds.map(preprocess, num_parallel_calls=AUTOTUNE)\
                   .cache()\
                   .batch(batch_size)\
                   .prefetch(buffer_size=AUTOTUNE)

    test_ds = test_ds.map(preprocess, num_parallel_calls=AUTOTUNE)\
                     .cache()\
                     .batch(batch_size)\
                     .prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds, info
