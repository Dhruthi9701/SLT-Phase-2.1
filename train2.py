import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.optimizers import Adam

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")

    # Load keypoint arrays from data folder
    X_train = np.load(os.path.join(data_dir, "X_train_keypoints.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train_keypoints.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test_keypoints.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test_keypoints.npy"))

    print("Loaded preprocessed keypoint data:")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    # One-hot encode labels
    num_classes = len(np.unique(y_train))
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    # Build MLP model
    model = Sequential([
        Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train
    #model.fit(X_train, y_train_cat, validation_data=(X_test, y_test_cat), epochs=100, batch_size=128, callbacks=[early_stop])
    model.fit(X_train, y_train_cat, validation_data=(X_test, y_test_cat), epochs=100, batch_size=128)

    # Evaluate
    loss, acc = model.evaluate(X_test, y_test_cat)
    print(f"Test accuracy: {acc:.4f}")

    # Predict classes for test set
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(20, 20))  # Larger figure for bigger boxes
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True, annot_kws={"size": 10})
    plt.xlabel('Predicted', fontsize=16)
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # Classification report
    # Example: Load or define your class names
    # class_names = ['A', 'B', 'C', ...]  # Replace with your actual class names

    # training images are in src/images for phrases/<class_name>/
    images_dir = os.path.join(data_dir, "images for phrases")
    class_names = sorted([d for d in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, d))])

    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=class_names))

    # Save model
    model.save("gesture_keypoint_mlp.keras")
    print("Model saved as gesture_keypoint_mlp.keras")

    # Fine-tuning: recompile with lower learning rate and train a few more epochs
    print("\nStarting fine-tuning...")
    model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train_cat, validation_data=(X_test, y_test_cat), epochs=10, batch_size=128)

    # Re-evaluate after fine-tuning
    loss, acc = model.evaluate(X_test, y_test_cat)
    print(f"Fine-tuned test accuracy: {acc:.4f}")

    # Save fine-tuned model
    model.save("gesture_keypoint_mlp_finetuned.keras")
    print("Fine-tuned model saved as gesture_keypoint_mlp_finetuned.keras")