import os
import cv2
import numpy as np
import random
import shutil

IMG_SIZE = (256, 256)  # Fixed size for images and depth maps
SPLIT_COUNT = 10  # Number of partitions

def collect_data_paths(data_path):
    """Collect all image and depth map paths."""
    image_paths = []
    depth_paths = []

    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith('.png'):
                img_path = os.path.join(root, file)
                
                # Collect RGB images (e.g., from 'image_02' folder)
                if 'image_02' in img_path:
                    image_paths.append(img_path)
                
                # Collect Depth maps (check if from 'groundtruth' or 'velodyne_raw' folder)
                elif 'groundtruth' in img_path or 'velodyne_raw' in img_path:
                    depth_paths.append(img_path)
    
    if len(image_paths) != len(depth_paths):
        print("Warning: Image and depth map counts do not match!")
    
    return image_paths, depth_paths

def shuffle_data(image_paths, depth_paths):
    """Shuffle image and depth map paths."""
    combined = list(zip(image_paths, depth_paths))
    random.shuffle(combined)
    image_paths[:], depth_paths[:] = zip(*combined)
    return image_paths, depth_paths

def preprocess_image(image):
    """Preprocess RGB image: Resize and Normalize."""
    image = cv2.resize(image, IMG_SIZE)  # Resize to fixed size
    image = image.astype(np.float32) / 255.0  # Normalize to [0,1]
    return image

def preprocess_depth(depth):
    """Preprocess Depth map: Resize and Normalize."""
    depth = cv2.resize(depth, IMG_SIZE, interpolation=cv2.INTER_NEAREST)
    depth = depth.astype(np.float32)
    if depth.max() > 0:
        depth = depth / np.max(depth)  # Normalize (scale to [0,1])
    return depth

def augment_data(image, depth):
    """Apply Data Augmentation (Flipping, Rotation, Brightness Adjustment, Translation, Scaling)."""
    if random.random() > 0.5:
        image = cv2.flip(image, 1)  # Horizontal flip
        depth = cv2.flip(depth, 1)

    if random.random() > 0.5:
        brightness_factor = 0.8 + 0.4 * random.random()
        image = np.clip(image * brightness_factor, 0, 1)

    # Add random rotation
    if random.random() > 0.5:
        angle = random.uniform(-10, 10)  # Random angle between -10 and 10 degrees
        M = cv2.getRotationMatrix2D((image.shape[1]//2, image.shape[0]//2), angle, 1)
        image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        depth = cv2.warpAffine(depth, M, (depth.shape[1], depth.shape[0]))

    # Add random translation
    if random.random() > 0.5:
        tx = random.randint(-10, 10)
        ty = random.randint(-10, 10)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        depth = cv2.warpAffine(depth, M, (depth.shape[1], depth.shape[0]))

    # Add random scaling
    if random.random() > 0.5:
        scale = random.uniform(0.8, 1.2)
        image = cv2.resize(image, None, fx=scale, fy=scale)
        depth = cv2.resize(depth, None, fx=scale, fy=scale)

    return image, depth

def load_and_preprocess_data(image_paths, depth_paths):
    """Generator function to load and preprocess images and depth maps."""
    for img_path, depth_path in zip(image_paths, depth_paths):
        image = cv2.imread(img_path)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        
        if image is None or depth is None:
            print(f"Skipping {img_path} or {depth_path} due to loading error.")
            continue
        
        image = preprocess_image(image)
        depth = preprocess_depth(depth)
        image, depth = augment_data(image, depth)
        
        yield ('image', image, img_path)
        yield ('depth', depth, depth_path)

def split_dataset(image_paths, depth_paths, output_dir, splits=SPLIT_COUNT):
    """Split dataset into partitions and save to output directory."""
    os.makedirs(output_dir, exist_ok=True)
    partition_size = len(image_paths) // splits
    
    for i in range(splits):
        part_dir = os.path.join(output_dir, f'partition_{i+1}')
        os.makedirs(part_dir, exist_ok=True)
        
        img_part = image_paths[i * partition_size:(i + 1) * partition_size]
        depth_part = depth_paths[i * partition_size:(i + 1) * partition_size]
        
        img_dir = os.path.join(part_dir, 'images')
        depth_dir = os.path.join(part_dir, 'depth_maps')
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)
        
        for img_path, depth_path in zip(img_part, depth_part):
            shutil.copy(img_path, img_dir)
            shutil.copy(depth_path, depth_dir)
    
        print(f"Partition {i+1} created with {len(img_part)} samples.")

if __name__ == "__main__":
    data_path = r'C:\Users\jeyas\OneDrive\Desktop\Depth_Estimation\data_depth_annotated'
    output_dir = r'C:\Users\jeyas\OneDrive\Desktop\Depth_Estimation\partitionData'
    
    # Step 1: Collect all file paths
    image_paths, depth_paths = collect_data_paths(data_path)
    
    # Step 2: Shuffle the data
    image_paths, depth_paths = shuffle_data(image_paths, depth_paths)
    
    # Step 3: Split dataset into 10 partitions
    split_dataset(image_paths, depth_paths, output_dir)
    
    print("Dataset splitting complete!")
