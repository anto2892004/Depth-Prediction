import torch
import cv2
import numpy as np
import torchvision.transforms as transforms

# Load MiDaS model
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")  # Change to "MiDaS_large" for better accuracy
midas.eval()

# Define preprocessing transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),  # Resize to match U-Net input size
    transforms.ToTensor(),
])

def get_midas_depth(image):
    """ Get initial depth estimation from MiDaS """
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = transform(img).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        depth = midas(img)
    
    depth = depth.squeeze().cpu().numpy()
    depth = cv2.resize(depth, (image.shape[1], image.shape[0]))  # Resize back to original size
    return depth

if __name__ == "__main__":
    image_path = r"C:\Users\jeyas\OneDrive\Desktop\unet\image4.jpg"  # Change this to your input image path
    image = cv2.imread(image_path)
    depth_map = get_midas_depth(image)

    cv2.imshow("MiDaS Depth", depth_map / depth_map.max())  # Normalize for display
    cv2.waitKey(0)
    cv2.destroyAllWindows()
