# SLT Phase 2.1 - Sign Language Translation

This project focuses on real-time Sign Language Translation (SLT) using computer vision and deep learning. It uses keypoint extraction to recognize gestures and translate them into text or speech.

## 🚀 Features
- **Keypoint Extraction:** Uses MediaPipe (or similar) to capture hand and body movements.
- **Data Collection:** Tools to record and append new sign language keypoints to the dataset.
- **Model Training:** Scripts to train a neural network on the collected keypoint data.
- **Live Prediction:** A real-time interface to predict signs using a webcam.

## 📂 Project Structure
- `data/`: Directory containing the collected keypoint datasets.
- `utils2.py`: Utility functions for processing landmarks and drawing on the screen.
- `append_new_keypoints.py`: Script to add new gesture data to your training set.
- `train2.py`: The main script to train the gesture recognition model.
- `live_predict2.py`: The script for running real-time inference via webcam.
- `test_camera.py`: A simple script to verify your camera setup.
