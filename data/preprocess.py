import os
import glob
import cv2
import mediapipe as mp
import numpy as np
from sklearn.model_selection import train_test_split

def get_images_and_labels(dataset_path):
    image_paths = []
    labels = []
    class_names = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
    for cls_name in class_names:
        cls_folder = os.path.join(dataset_path, cls_name)
        for ext in ("*.jpg", "*.png", "*.jpeg"):
            for img_file in glob.glob(os.path.join(cls_folder, ext)):
                image_paths.append(img_file)
                labels.append(class_to_idx[cls_name])
    return image_paths, labels, class_names

def extract_all_keypoints(image_path, holistic, segmentor=None, backgrounds=None):
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image {image_path}. Skipping.")
            return None

        # Background augmentation is now commented out and not used

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = holistic.process(img_rgb)

        # Right hand (21x3)
        right_hand = np.zeros((21, 3), dtype=np.float32)
        if results.right_hand_landmarks:
            for i, lm in enumerate(results.right_hand_landmarks.landmark):
                right_hand[i] = [lm.x, lm.y, lm.z]

        # Left hand (21x3)
        left_hand = np.zeros((21, 3), dtype=np.float32)
        if results.left_hand_landmarks:
            for i, lm in enumerate(results.left_hand_landmarks.landmark):
                left_hand[i] = [lm.x, lm.y, lm.z]

        # Pose (33x4)
        pose = np.zeros((33, 4), dtype=np.float32)
        if results.pose_landmarks:
            for i, lm in enumerate(results.pose_landmarks.landmark):
                pose[i] = [lm.x, lm.y, lm.z, lm.visibility]

        # Face (468x3)
        face = np.zeros((468, 3), dtype=np.float32)
        if results.face_landmarks:
            for i, lm in enumerate(results.face_landmarks.landmark):
                face[i] = [lm.x, lm.y, lm.z]

        # Flatten all and concatenate
        feature_vector = np.concatenate([
            right_hand.flatten(),
            left_hand.flatten(),
            pose.flatten(),
            face.flatten()
        ])
        return feature_vector
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def extract_keypoints_dataset(image_paths, labels):
    features = []
    valid_labels = []
    mp_holistic = mp.solutions.holistic
    mp_selfie = mp.solutions.selfie_segmentation
    holistic = mp_holistic.Holistic(static_image_mode=True)
    segmentor = mp_selfie.SelfieSegmentation(model_selection=1)

    # Load backgrounds
    #current_dir = os.path.dirname(os.path.abspath(__file__))
    #backgrounds_dir = os.path.join(current_dir, "..", "backgrounds")
    #backgrounds = []
    #for ext in ("*.jpg", "*.png", "*.jpeg"):
    #    backgrounds.extend(glob.glob(os.path.join(backgrounds_dir, ext)))
    #if not backgrounds:
    #    print("Warning: No backgrounds found for augmentation.")

    for idx, img_path in enumerate(image_paths):
        keypoints = extract_all_keypoints(img_path, holistic)
        if keypoints is not None:
            features.append(keypoints)
            valid_labels.append(labels[idx])
        if idx % 50 == 0:
            print(f"Processed {idx+1}/{len(image_paths)} images")
    
    holistic.close()
    segmentor.close()
    return np.array(features), np.array(valid_labels)

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_path = os.path.join(current_dir, "images for phrases")

    images, labels, class_names = get_images_and_labels(target_path)
    print(f"Found {len(images)} images belonging to {len(class_names)} classes.")
    print("Class folders found:", class_names)

    # Extract keypoints with error handling
    X_keypoints, y = extract_keypoints_dataset(images, labels)

    # Split and save
    X_train, X_test, y_train, y_test = train_test_split(X_keypoints, y, test_size=0.2, random_state=42)

    print("Saving .npy files to:", current_dir)
    np.save(os.path.join(current_dir, "X_train_keypoints.npy"), X_train)
    np.save(os.path.join(current_dir, "y_train_keypoints.npy"), y_train)
    np.save(os.path.join(current_dir, "X_test_keypoints.npy"), X_test)
    np.save(os.path.join(current_dir, "y_test_keypoints.npy"), y_test)
    print("Keypoint arrays saved as .npy files.")
