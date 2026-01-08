import cv2
import numpy as np
import mediapipe as mp
import os
import logging
import time

def save_model(model, filepath):
    model.save(filepath)

def load_model(filepath):
    from tensorflow.keras.models import load_model
    return load_model(filepath)

def log_message(message):
    logging.basicConfig(level=logging.INFO)
    logging.info(message)

def extract_keypoints_from_image(frame, holistic):
    try:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(img_rgb)

        if not results.right_hand_landmarks and not results.left_hand_landmarks:
            return None

        right_hand = np.zeros((21, 3), dtype=np.float32)
        if results.right_hand_landmarks:
            for i, lm in enumerate(results.right_hand_landmarks.landmark):
                right_hand[i] = [lm.x, lm.y, lm.z]

        left_hand = np.zeros((21, 3), dtype=np.float32)
        if results.left_hand_landmarks:
            for i, lm in enumerate(results.left_hand_landmarks.landmark):
                left_hand[i] = [lm.x, lm.y, lm.z]

        pose = np.zeros((33, 4), dtype=np.float32)
        if results.pose_landmarks:
            for i, lm in enumerate(results.pose_landmarks.landmark):
                pose[i] = [lm.x, lm.y, lm.z, lm.visibility]

        face = np.zeros((468, 3), dtype=np.float32)
        if results.face_landmarks:
            for i, lm in enumerate(results.face_landmarks.landmark):
                face[i] = [lm.x, lm.y, lm.z]

        feature_vector = np.concatenate([
            right_hand.flatten(),
            left_hand.flatten(),
            pose.flatten(),
            face.flatten()
        ])
        return feature_vector
    except Exception as e:
        log_message(f"Error extracting keypoints from frame: {e}")
        return None



def save_live_error_image(image, error_dir="live_errors"):
    if not os.path.exists(error_dir):
        os.makedirs(error_dir)
    existing = [f for f in os.listdir(error_dir) if f.startswith("error_") and f.endswith(".png")]
    numbers = [int(f.split("_")[1].split(".")[0]) for f in existing if f.split("_")[1].split(".")[0].isdigit()]
    next_num = max(numbers) + 1 if numbers else 1
    filename = f"error_{next_num}.png"
    filepath = os.path.join(error_dir, filename)
    cv2.imwrite(filepath, image)
    print(f"Saved error image: {filepath}")

def live_prediction_loop(model, class_names, holistic, cap):
    last_keypoints = None
    stagnant_start_time = None
    prediction_sentence = ""
    SPACE_THRESHOLD = 1

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        keypoints = extract_keypoints_from_image(frame, holistic)
        display_text = "Detecting..."

        if keypoints is not None:
            keypoints = keypoints.flatten()
            if last_keypoints is not None:
                movement = np.linalg.norm(keypoints - last_keypoints)
                if movement < 1e-3:
                    if stagnant_start_time is None:
                        stagnant_start_time = time.time()
                    elif time.time() - stagnant_start_time > SPACE_THRESHOLD:
                        if not prediction_sentence.endswith(" "):
                            prediction_sentence += " "
                            print("Space added due to no movement.")
                        stagnant_start_time = None
                else:
                    stagnant_start_time = None
            last_keypoints = keypoints.copy()

            keypoints = keypoints.reshape(1, -1)
            pred = model.predict(keypoints)
            gesture_idx = int(np.argmax(pred))
            confidence = float(np.max(pred))
            phrase = class_names[gesture_idx] if gesture_idx < len(class_names) else "Unknown"
            display_text = f"{phrase} ({confidence:.2f})"
            if phrase != "Unknown" and confidence > 0.5:
                prediction_sentence += phrase + " "
            if confidence < 0.5:
                save_live_error_image(frame)
            cv2.putText(frame, display_text, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No keypoints detected", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.putText(frame, f"Sentence: {prediction_sentence}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 0), 2)
        cv2.imshow("Live Keypoint Prediction", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 13:
            prediction_sentence = ""
            print("New sentence started.")

    cap.release()
    cv2.destroyAllWindows()
