
# Depth Prediction – U-Net + MiDaS Pipeline

A **monocular depth estimation pipeline** built with **MiDaS** and **U-Net** for improved accuracy and sharper outputs on the **KITTI dataset**.  
This project uses **novel preprocessing techniques** such as **shuffled patch augmentation** to enhance spatial learning and robustness.

---

## ✨ Key Features
- **Two-stage architecture**:
  1. **MiDaS** generates coarse depth estimations.
  2. **U-Net** refines and sharpens the predictions.
- **Novel preprocessing**: Shuffled patch augmentation improves robustness to spatial variations.
- Optimized for the **KITTI dataset**.
- Supports both **image** and **video** depth estimation.

---

## 🛠 Pipeline Flow

```

RGB Image
│
▼
MiDaS Model  ──►  Initial Depth Map
│
▼
Shuffled Patch Augmentation
│
▼
U-Net Refinement
│
▼
Final Depth Map

````

---

## 📦 Installation
```bash
git clone https://github.com/anto2892004/Depth-Prediction.git
cd Depth-Prediction
pip install -r requirements.txt
````

---

## 🚀 Usage

### 1️⃣ Training on KITTI

```bash
python train.py \
  --data-path /path/to/KITTI \
  --epochs 30 \
  --batch-size 8
```

### 2️⃣ Evaluation

```bash
python eval.py \
  --data-path /path/to/KITTI \
  --ckpt model.pth
```

### 3️⃣ Inference

**Image:**

```bash
python infer_image.py \
  --input image.jpg \
  --ckpt model.pth
```

**Video:**

```bash
python infer_video.py \
  --input video.mp4 \
  --ckpt model.pth
```

---

## 📂 Project Structure

```
src/
  ├── preprocess.py     # Shuffling & augmentation methods
  ├── midas_model.py    # MiDaS depth estimation
  ├── unet_model.py     # U-Net refinement
  ├── train.py          # Training script
  ├── eval.py           # Evaluation script
  ├── infer_image.py    # Image inference
  ├── infer_video.py    # Video inference
requirements.txt
```

---



