import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from utils2 import live_prediction_loop
import os

model = load_model("gesture_keypoint_mlp_finetuned.keras")

class_names = sorted(os.listdir("data/images for phrases"))

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

live_prediction_loop(model, class_names, holistic, cap)

holistic.close()
cap.release()
cv2.destroyAllWindows()
