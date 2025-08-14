import numpy as np
from utils2 import extract_keypoints_from_image
import mediapipe as mp
import os
import cv2

# Load existing data
try:
    X_train = np.load("data/X_train_keypoints.npy")
    y_train = np.load("data/y_train_keypoints.npy")
    print("Loaded X_train shape:", X_train.shape)
    print("Loaded y_train shape:", y_train.shape)
except Exception as e:
    print("Error loading .npy files:", e)
    exit()

error_folder = "live_errors"
processed_file = "processed_images.txt"

# Load class names (phrases) in sorted order
class_names = sorted(os.listdir("data/images for phrases"))

# Track processed images
if os.path.exists(processed_file):
    with open(processed_file, "r") as f:
        processed_images = set(line.strip() for line in f)
else:
    processed_images = set()

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=True)

new_keypoints = []
valid_new_labels = []
newly_processed = []

image_files = [fname for fname in os.listdir(error_folder) if fname.lower().endswith((".jpg", ".png", ".jpeg"))]
print(f"Found {len(image_files)} image files in {error_folder}")

unprocessed = [f for f in image_files if os.path.join(error_folder, f) not in processed_images]
print(f"{len(unprocessed)} images to process")

for fname in image_files:
    img_path = os.path.join(error_folder, fname)
    if fname in processed_images:
        print(f"Skipping already processed: {img_path}")
        continue
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read {img_path}")
        continue
    cv2.imshow("Label this image (type phrase and press Enter)", img)
    print(f"Label for: {img_path}")
    cv2.waitKey(1)  # Give time for window to render

    phrase = input("Enter the phrase this gesture belongs to (or type 'skip' to skip): ").strip()
    cv2.destroyAllWindows()

    if phrase.lower() == "skip":
        print("Skipping.")
        continue

    phrase_normalized = phrase.lower()
    class_names_normalized = [c.lower() for c in class_names]

    if phrase_normalized in class_names_normalized:
        label_idx = class_names_normalized.index(phrase_normalized)
        kp = extract_keypoints_from_image(img, holistic)
        if kp is not None:
            new_keypoints.append(kp)
            valid_new_labels.append(label_idx)
            newly_processed.append(fname)
            print(f"Labeled and processed {img_path} as '{class_names[label_idx]}' (index {label_idx})")
            os.remove(img_path)
            print(f"Deleted {img_path}")
        else:
            print(f"Keypoints not found for {img_path}")
    else:
        print(f"Phrase '{phrase}' not found in class_names. Skipping.")

holistic.close()

# Convert to numpy arrays and append
if new_keypoints:
    new_keypoints = np.array(new_keypoints)
    valid_new_labels = np.array(valid_new_labels)

    X_train_updated = np.concatenate([X_train, new_keypoints], axis=0)
    y_train_updated = np.concatenate([y_train, valid_new_labels], axis=0)

    np.save("data/X_train_keypoints.npy", X_train_updated)
    np.save("data/y_train_keypoints.npy", y_train_updated)
    print(f"Appended {len(new_keypoints)} new keypoints to training data.")

    # Update processed images file
    with open(processed_file, "a") as f:
        for img_path in newly_processed:
            f.write(img_path + "\n")
else:
    print("No new images were processed.")


