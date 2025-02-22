# Filename: preprocess_fer2013.py
# Location: facial_mood_detector_cloud/preprocess_fer2013.py
#
# USAGE:
#   1. Open a command prompt in your project folder (facial_mood_detector_cloud).
#   2. Run: python preprocess_fer2013.py
#   3. Check 'facial_mood_detector_cloud/dataset/preprocessed/' for .npy files.

import os
import cv2
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical # type: ignore

def ensure_directory_exists(dir_path):
    """Create the directory if it doesn't exist."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def convert_pixels_to_images(df):
    """
    Takes a DataFrame with columns ['emotion', 'pixels', 'Usage'].
    Returns:
      - images (NumPy array of shape [N, 48, 48])
      - labels (NumPy array of shape [N])
    """
    images = []
    labels = []

    for _, row in df.iterrows():
        emotion_label = row['emotion']  # numeric class: 0..6
        pixel_str = row['pixels']      # space-separated string of 2304 pixel values

        # Convert pixel string to list of ints
        pixel_values = list(map(int, pixel_str.split()))
        # Reshape into 48x48
        img_array = np.array(pixel_values).reshape(48, 48).astype('uint8')

        images.append(img_array)
        labels.append(emotion_label)

    images = np.array(images)
    labels = np.array(labels)
    return images, labels

def preprocess_images(images):
    """
    Takes grayscale images of shape (N, 48, 48).
    - Resizes to 224x224
    - Converts to 3-channel (BGR)
    - Normalizes pixel values to [0,1]
    Returns an array of shape (N, 224, 224, 3).
    """
    processed = []
    for img in images:
        # Resize 48x48 -> 224x224
        img_resized = cv2.resize(img, (224, 224))

        # Convert grayscale -> BGR (3 channels)
        img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)

        # Normalize to [0, 1]
        img_normalized = img_bgr / 255.0

        processed.append(img_normalized)

    return np.array(processed, dtype=np.float32)

def main():
    """
    Main function that:
      1) Loads fer2013.csv
      2) Splits it by Usage (train, val, test)
      3) Converts pixel strings into 48x48 arrays
      4) Resizes to 224x224, normalizes, converts to 3-channel
      5) One-hot encodes labels
      6) Saves .npy files
    """
    # Adjust paths if needed
    fer_csv_path = os.path.join("dataset", "fer2013.csv")
    output_dir   = os.path.join("dataset", "preprocessed")

    # Ensure output directory exists
    ensure_directory_exists(output_dir)

    # 1. Load CSV with pandas
    print("[INFO] Loading FER2013 CSV...")
    fer_data = pd.read_csv(fer_csv_path)

    # 2. Split by Usage
    print("[INFO] Splitting DataFrame into train/val/test...")
    train_data = fer_data[fer_data['Usage'] == 'Training']
    val_data   = fer_data[fer_data['Usage'] == 'PublicTest']
    test_data  = fer_data[fer_data['Usage'] == 'PrivateTest']

    # 3. Convert pixel strings -> 48x48 arrays
    print("[INFO] Converting pixel strings to 48x48 arrays...")
    X_train_raw, y_train = convert_pixels_to_images(train_data)
    X_val_raw,   y_val   = convert_pixels_to_images(val_data)
    X_test_raw,  y_test  = convert_pixels_to_images(test_data)

    # 4. Resize, convert to 3 channels, normalize
    print("[INFO] Preprocessing images (resizing, normalizing, 3 channels)...")
    X_train = preprocess_images(X_train_raw)
    X_val   = preprocess_images(X_val_raw)
    X_test  = preprocess_images(X_test_raw)

    # 5. One-hot encode emotion labels
    #    If your dataset has 7 classes (0..6), you can set num_emotions=7.
    #    Or auto-detect by looking at the maximum label.
    num_emotions = y_train.max() + 1  # auto
    print(f"[INFO] Detected {num_emotions} emotion classes.")
    y_train_cat = to_categorical(y_train, num_classes=num_emotions)
    y_val_cat   = to_categorical(y_val,   num_classes=num_emotions)
    y_test_cat  = to_categorical(y_test,  num_classes=num_emotions)

    # 6. Save npy files
    print("[INFO] Saving .npy files to:", output_dir)
    np.save(os.path.join(output_dir, "fer2013_train_images.npy"), X_train)
    np.save(os.path.join(output_dir, "fer2013_val_images.npy"),   X_val)
    np.save(os.path.join(output_dir, "fer2013_test_images.npy"),  X_test)

    np.save(os.path.join(output_dir, "fer2013_train_labels.npy"), y_train_cat)
    np.save(os.path.join(output_dir, "fer2013_val_labels.npy"),   y_val_cat)
    np.save(os.path.join(output_dir, "fer2013_test_labels.npy"),  y_test_cat)

    print("[INFO] Preprocessing complete.")
    print("[INFO] Shapes:")
    print("  X_train:", X_train.shape, "y_train:", y_train_cat.shape)
    print("  X_val:  ", X_val.shape,   "y_val:  ", y_val_cat.shape)
    print("  X_test: ", X_test.shape,  "y_test: ", y_test_cat.shape)

if __name__ == "__main__":
    main()
