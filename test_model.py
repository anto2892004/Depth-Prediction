import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load the trained model
MODEL_PATH = "depth_estimation_model.h5"
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Function to preprocess the uploaded image
def preprocess_image(img_path, target_size=(256, 256)):
    """
    Preprocesses the uploaded image to the format the model expects.
    
    Args:
    - img_path (str): Path to the uploaded image.
    - target_size (tuple): Target size to resize the image.
    
    Returns:
    - np.ndarray: Preprocessed image ready for prediction.
    """
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize if required
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to make predictions
def predict_depth(img_path):
    """
    Predicts depth map using the trained model.
    
    Args:
    - img_path (str): Path to the image.
    
    Returns:
    - np.ndarray: Predicted depth map.
    """
    preprocessed_image = preprocess_image(img_path)
    predicted_depth_map = model.predict(preprocessed_image)
    return predicted_depth_map.squeeze()

# Function to visualize and save results
def save_results(img_path, predicted_depth_map, output_image_path, output_csv_path):
    """
    Visualizes, saves depth map as an image, and stores depth values in a CSV file.
    
    Args:
    - img_path (str): Path to the uploaded image.
    - predicted_depth_map (np.ndarray): The predicted depth map.
    - output_image_path (str): Path to save the depth map image.
    - output_csv_path (str): Path to save the depth values as CSV.
    """
    # Load original image
    original_image = cv2.imread(img_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Save depth map as an image
    plt.imsave(output_image_path, predicted_depth_map, cmap='plasma')

    # Convert depth values to CSV format
    pd.DataFrame(predicted_depth_map).to_csv(output_csv_path, index=False, header=False)

    # Plot original and depth map
    plt.figure(figsize=(10, 5))
    
    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis('off')

    # Depth map
    plt.subplot(1, 2, 2)
    plt.imshow(predicted_depth_map, cmap='plasma')
    plt.title("Predicted Depth Map")
    plt.axis('off')

    plt.show()
    print(f"✅ Depth map saved as image: {output_image_path}")
    print(f"✅ Depth values saved as CSV: {output_csv_path}")

# Main function to test the model
def test_model(img_path):
    """
    Runs the model, saves results, and displays predictions.
    
    Args:
    - img_path (str): Path to the image.
    """
    print("🔄 Making prediction for the uploaded image...")
    
    # Predict depth
    predicted_depth_map = predict_depth(img_path)

    # Define output file names
    output_image_path = "output_depth.png"
    output_csv_path = "depth_results.csv"

    # Save results
    save_results(img_path, predicted_depth_map, output_image_path, output_csv_path)

# Set image path and run
image_path = r"C:\unet\image4.jpg"

if __name__ == "__main__":
    test_model(image_path)
