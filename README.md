Depth-Estimation
Depth Estimation using MiDaS and U-Net

Overview

This repository presents a monocular depth estimation model that integrates MiDaS and U-Net in a pipeline to generate accurate depth maps. Depth estimation plays a crucial role in various computer vision applications, including autonomous driving, robotics, and augmented reality. By leveraging the strengths of both MiDaS and U-Net, the model achieves high precision in depth predictions while being optimized for real-time deployment.

Features

Pipeline Integration: The model combines MiDaS for initial depth estimation and U-Net for refinement, enhancing the accuracy and sharpness of depth maps. Dataset Utilization: The model is trained and validated using the KITTI dataset, a widely used benchmark for depth estimation tasks. High Accuracy: The combined MiDaS + U-Net approach achieves an accuracy of 96% on the processed dataset. Edge AI Compatibility: Optimized for deployment on edge devices, ensuring efficient depth estimation within a compact model size of 10MB. Real-time Processing: The pipeline is designed for fast inference, making it suitable for applications requiring real-time depth estimation.

Model Architecture

The depth estimation pipeline consists of the following components: MiDaS Preprocessing: The MiDaS model generates an initial depth estimate from monocular images, providing a coarse depth map. U-Net Refinement: The output from MiDaS is passed through a U-Net model, which refines depth predictions by preserving local features and improving depth accuracy. Training Pipeline: The U-Net model is trained on a custom dataset derived from KITTI, incorporating data augmentation, image normalization, and optimized loss functions for improved performance. Deployment: The final model is optimized for Edge AI devices, ensuring a balance between accuracy and computational efficiency.
