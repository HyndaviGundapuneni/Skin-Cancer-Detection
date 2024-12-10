# Libraries and Google Drive setup
from google.colab import drive
drive.mount('/content/gdrive')

# Unzip the dataset
!unzip "/content/gdrive/MyDrive/SkinCancerDataset.zip" > /dev/null

# Import the required libraries
import pathlib
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import load_img
import seaborn as sns
from glob import glob

# Paths for train and test datasets
train_dir = pathlib.Path("/content/Skin cancer ISIC The International Skin Imaging Collaboration/Train/")
test_dir = pathlib.Path("/content/Skin cancer ISIC The International Skin Imaging Collaboration/Test/")

# Count images in Train and Test directories
def count_images(directory):
    return len(list(directory.glob('*/*.jpg')))

print("Train Images:", count_images(train_dir))
print("Test Images:", count_images(test_dir))

# Function to visualize classes
def visualize_classes(data_dir, class_names):
    class_files = {cls: list(data_dir.glob(f"{cls}/*"))[:1] for cls in class_names}
    plt.figure(figsize=(12, 8))
    for idx, (cls, paths) in enumerate(class_files.items()):
        plt.subplot(3, 3, idx + 1)
        img = load_img(str(paths[0]), target_size=(180, 180))
        plt.imshow(img)
        plt.title(cls)
        plt.axis("off")
    plt.show()

# Load dataset and fetch class names
image_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir, batch_size=32, image_size=(180, 180), label_mode='categorical', seed=123
)
class_names = image_dataset.class_names
visualize_classes(train_dir, class_names)

# Visualize class distribution
def visualize_distribution(directory):
    df = pd.DataFrame([
        {"Class": subdir.name, "No. of Images": len(list(subdir.glob('*')))}
        for subdir in directory.iterdir() if subdir.is_dir()
    ])
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, y="Class", x="No. of Images")
    plt.title("Class Distribution in Training Dataset")
    plt.show()
visualize_distribution(train_dir)

# Augment the dataset to handle class imbalance
!pip install Augmentor
def augment_data(directory, class_names):
    for cls in class_names:
        aug = Augmentor.Pipeline(str(directory / cls))
        aug.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
        aug.sample(500)

augment_data(train_dir, class_names)
print("Images after augmentation:", count_images(train_dir))

# Prepare Train and Validation datasets
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir, batch_size=32, image_size=(180, 180), label_mode='categorical',
    seed=123, subset="training", validation_split=0.2
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir, batch_size=32, image_size=(180, 180), label_mode='categorical',
    seed=123, subset="validation", validation_split=0.2
)
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# Model architecture
model = Sequential([
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(180, 180, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.5),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.25),
    layers.Dense(len(class_names), activation='softmax')
])
model.summary()

# Compile and train the model
model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])
callbacks = [
    ModelCheckpoint("model.h5", monitor="val_accuracy", save_best_only=True, verbose=1),
    EarlyStopping(monitor="val_accuracy", patience=5, verbose=1)
]
history = model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=callbacks)

# Plot training metrics
def plot_training(history):
    epochs_range = range(len(history.history['accuracy']))
    plt.figure(figsize=(12, 5))
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history.history['accuracy'], label="Training")
    plt.plot(epochs_range, history.history['val_accuracy'], label="Validation")
    plt.title("Model Accuracy")
    plt.legend()
    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history.history['loss'], label="Training")
    plt.plot(epochs_range, history.history['val_loss'], label="Validation")
    plt.title("Model Loss")
    plt.legend()
    plt.show()
plot_training(history)

# Test model prediction
def predict_sample(image_path, model, class_names):
    img = load_img(image_path, target_size=(180, 180))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    pred_class = class_names[np.argmax(pred)]
    plt.imshow(img)
    plt.title(f"Predicted: {pred_class}")
    plt.axis("off")
    plt.show()

test_sample = list(test_dir.glob(f"{class_names[1]}/*"))[-1]
predict_sample(str(test_sample), model, class_names)
