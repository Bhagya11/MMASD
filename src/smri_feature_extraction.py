import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, GlobalMaxPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_vgg
from tensorflow.keras.applications.resnet import preprocess_input as preprocess_input_resnet

def create_cnn_model():
    """Creates a custom CNN model for feature extraction."""
    model = Sequential([
        Conv2D(30, kernel_size=(3, 3), input_shape=(224, 224, 3)),
        MaxPool2D(),
        Conv2D(25, kernel_size=(3, 3)),
        MaxPool2D(),
        Conv2D(15, kernel_size=(3, 3)),
        MaxPool2D(),
        Conv2D(10, kernel_size=(3, 3)),
        MaxPool2D(),
        Flatten(),
        Dense(100, activation="relu"),
        Dense(170, activation="relu")
    ])
    model.summary()
    return model

def create_resnet_model():
    """Creates a ResNet-based model for feature extraction."""
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    model = Sequential([
        base_model,
        GlobalMaxPool2D(),
        Dense(128, activation='relu'),
        Dense(170, activation='softmax')
    ])
    model.summary()
    return model

def create_vgg_model():
    """Creates a VGG16-based model for feature extraction."""
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    model = Sequential([
        base_model,
        GlobalMaxPool2D(),
        Dense(128, activation='relu'),
        Dense(256, activation='relu'),
        Dense(170, activation='softmax')
    ])
    model.summary()
    return model

def extract_features(image_folder, model, preprocessor, output_csv_path):
    """
    Loads images, extracts features using the provided model, and saves them to a CSV file.

    Args:
        image_folder (str): Path to the folder containing the images.
        model (tf.keras.Model): The model to use for feature extraction.
        preprocessor (function): The preprocessing function for the model.
        output_csv_path (str): The path to save the output CSV file.
    """
    images = []
    image_paths = []
    
    for subdir in ["Autism", "NonAutism"]:
        subdir_path = os.path.join(image_folder, subdir)
        if os.path.isdir(subdir_path):
            for filename in os.listdir(subdir_path):
                img_path = os.path.join(subdir_path, filename)
                img = load_img(img_path, target_size=(224, 224))
                img = img_to_array(img)
                img = preprocessor(img)
                images.append(img)
                image_paths.append(img_path)

    images = tf.convert_to_tensor(images)
    features = model.predict(images)
    
    feature_df = pd.DataFrame(features)
    
    # Extract subject ID from path
    subject_ids = [int(path.split('_')[-1].split('.')[0]) for path in image_paths]
    feature_df['subject'] = subject_ids
    
    feature_df = feature_df.sort_values("subject").reset_index(drop=True)
    feature_df.to_csv(output_csv_path, index=False)
    print(f"Features saved to {output_csv_path}")

def extract_smri_features(image_folder):
    """
    Orchestrates the feature extraction process for all three models (CNN, ResNet, VGG).

    Args:
        image_folder (str): The folder containing the sMRI images.
    """
    # CNN
    print("Extracting features with custom CNN...")
    cnn_model = create_cnn_model()
    extract_features(image_folder, cnn_model, preprocess_input_vgg, 'cnn.csv')

    # ResNet
    print("\nExtracting features with ResNet50...")
    resnet_model = create_resnet_model()
    extract_features(image_folder, resnet_model, preprocess_input_resnet, 'resnet.csv')

    # VGG16
    print("\nExtracting features with VGG16...")
    vgg_model = create_vgg_model()

    extract_features(image_folder, vgg_model, preprocess_input_vgg, 'vgg.csv')
