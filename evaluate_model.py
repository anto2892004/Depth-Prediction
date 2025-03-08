import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Dataset path where the partitions are stored
base_partition_path = r"C:\Users\jeyas\OneDrive\Desktop\Depth_Estimation\partitionData"

# Model path
model_path = r"C:\Users\jeyas\OneDrive\Desktop\Depth_Estimation\depth_estimation_model.h5"

# Define image size (update based on your model input size)
IMG_HEIGHT = 256
IMG_WIDTH = 256  # Adjusted to match the model input size

def parse_image(image_path, depth_path):
    """Helper function to parse images and depth maps."""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = image / 255.0  # Normalize image
    
    depth_map = tf.io.read_file(depth_path)
    depth_map = tf.image.decode_jpeg(depth_map, channels=3)
    depth_map = tf.image.resize(depth_map, [IMG_HEIGHT, IMG_WIDTH])
    depth_map = depth_map / 255.0  # Normalize depth map
    
    return image, depth_map

def load_data_in_batches(partition_paths, batch_size=16):
    """Load images and depth maps from given partition paths in batches."""
    image_paths = []
    depth_paths = []

    # Collect image and depth paths from all partitions
    for partition in partition_paths:
        images_folder = os.path.join(partition, "images")
        depth_folder = os.path.join(partition, "depth_maps")
        
        if not os.path.exists(images_folder) or not os.path.exists(depth_folder):
            print(f"❌ Missing folders in partition: {partition}")
            continue
        
        for filename in sorted(os.listdir(images_folder)):
            image_paths.append(os.path.join(images_folder, filename))
            depth_paths.append(os.path.join(depth_folder, filename))
    
    # Create a TensorFlow dataset from the file paths
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, depth_paths))
    dataset = dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)  # Parse the images and depth maps
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)  # Prepare data while model is training
    
    return dataset

def calculate_accuracy(y_true, y_pred, threshold=0.1):
    """Calculate the percentage of predictions within the given threshold."""
    abs_error = np.abs(y_true - y_pred)
    correct_predictions = np.sum(abs_error < threshold)
    accuracy = (correct_predictions / abs_error.size) * 100
    return accuracy

# Load test data
test_partitions = [
    os.path.join(base_partition_path, "partition_9"),
    os.path.join(base_partition_path, "partition_10")
]

# Load the trained model
print(f"📥 Loading model from: {model_path}")
model = load_model(model_path)

# Create the dataset
dataset = load_data_in_batches(test_partitions, batch_size=16)

# Initialize variables for tracking evaluation results
total_loss = 0
total_mae = 0
total_accuracy = 0
total_samples = 0

# Evaluate model in batches
for images, depth_maps in dataset:
    print(f"🔍 Evaluating batch with {images.shape[0]} samples...")

    # Evaluate the model on the current batch
    loss, mae = model.evaluate(images, depth_maps, verbose=0)
    y_pred = model.predict(images)

    # Calculate accuracy for the current batch
    accuracy = calculate_accuracy(depth_maps, y_pred, threshold=0.1)

    # Accumulate results for final averages
    total_loss += loss * len(images)
    total_mae += mae * len(images)
    total_accuracy += accuracy * len(images)
    total_samples += len(images)

    # Optionally, print batch evaluation
    print(f"Batch Results - Loss: {loss:.4f}, MAE: {mae:.4f}, Accuracy: {accuracy:.2f}%")

# Calculate overall averages
avg_loss = total_loss / total_samples
avg_mae = total_mae / total_samples
avg_accuracy = total_accuracy / total_samples

# Print final evaluation results
print(f"📊 Final Evaluation Results:")
print(f"🔹 Average Loss: {avg_loss:.4f}")
print(f"🔹 Average MAE: {avg_mae:.4f}")
print(f"🔹 Average Accuracy: {avg_accuracy:.2f}%")
