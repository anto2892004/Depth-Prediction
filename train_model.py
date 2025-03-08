import os
import time
import numpy as np
import tensorflow as tf
import cv2  # Make sure to import OpenCV for resizing
from sklearn.model_selection import train_test_split
from load_data import collect_data_paths, shuffle_data, load_and_preprocess_data
from build_model import unet

# Constants
BATCH_SIZE = 8
EPOCHS_PER_PARTITION = 5  # Number of epochs per partition
PARTITIONS_DIR = r"C:\Users\jeyas\OneDrive\Desktop\Depth_Estimation\partitionData"
MODEL_SAVE_PATH = "depth_estimation_model.h5"

def prepare_data(image_paths, depth_paths):
    """Load, preprocess, and convert data to NumPy arrays."""
    images, depths = [], []
    
    for data_type, data, path in load_and_preprocess_data(image_paths, depth_paths):
        if data_type == 'image':
            # Resize image to a fixed size (256x256)
            if data.shape[:2] != (256, 256):  # Check if the image is not already 256x256
                data = cv2.resize(data, (256, 256))
            print(f"Image shape after resize: {data.shape}")
            images.append(data)
        elif data_type == 'depth':
            # Resize depth map to a fixed size (256x256)
            if data.shape[:2] != (256, 256):  # Check if the depth map is not already 256x256
                data = cv2.resize(data, (256, 256))
            print(f"Depth shape after resize: {data.shape}")
            depths.append(data)

    # Convert lists to numpy arrays
    try:
        images = np.array(images)
        depths = np.array(depths)
    except ValueError as e:
        print("Error while converting to numpy arrays:", e)
        print(f"Images shape: {len(images)}")
        print(f"Depths shape: {len(depths)}")
        # Check individual shapes for debugging
        for i, (img, dep) in enumerate(zip(images, depths)):
            print(f"Image {i} shape: {img.shape}, Depth {i} shape: {dep.shape}")
        raise e  # Re-raise the exception after printing debug info

    return images, depths

# Load model (if exists, continue training)
if os.path.exists(MODEL_SAVE_PATH):
    print("🔄 Loading existing model...")
    model = tf.keras.models.load_model(MODEL_SAVE_PATH, compile=False)
else:
    print("🚀 Creating new model...")
    model = unet()

# Compile with optimized settings
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, decay=1e-6),
              loss=tf.keras.losses.Huber(delta=1.0),
              metrics=["mae"])

# Train model on each partition sequentially
partition_dirs = sorted(os.listdir(PARTITIONS_DIR))

for partition in partition_dirs:
    partition_path = os.path.join(PARTITIONS_DIR, partition)
    print(f"\n📂 Training on {partition}...")

    image_paths = sorted([os.path.join(partition_path, "images", f) for f in os.listdir(os.path.join(partition_path, "images"))])
    depth_paths = sorted([os.path.join(partition_path, "depth_maps", f) for f in os.listdir(os.path.join(partition_path, "depth_maps"))])

    # Shuffle data before training
    image_paths, depth_paths = shuffle_data(image_paths, depth_paths)

    # Load and preprocess data
    X, y = prepare_data(image_paths, depth_paths)

    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model on this partition
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=EPOCHS_PER_PARTITION,
                        batch_size=BATCH_SIZE)

    # Save the model after training on each partition
    model.save(MODEL_SAVE_PATH)
    print(f"✅ Model saved after training on {partition}")

    # 30-second delay before loading the next partition
    print("⏳ Waiting 30 seconds before training on next partition...\n")
    time.sleep(30)

print("🎉 Training complete! Model saved as", MODEL_SAVE_PATH)
