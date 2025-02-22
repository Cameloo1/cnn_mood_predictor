"""
Filename: train_fer2013_generator.py

This script demonstrates training a simple CNN on FER2013
by streaming images from the CSV (Method 5: On-the-fly data generator).

REQUIREMENTS:
  - pip install pandas numpy opencv-python tensorflow

USAGE:
  1. Place this file in your 'facial_mood_detector_cloud' folder.
  2. Make sure fer2013.csv is in: dataset/fer2013/fer2013.csv
  3. Open a terminal or PowerShell in 'facial_mood_detector_cloud'.
  4. Run: python train_fer2013_generator.py
"""

import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.utils import Sequence, to_categorical # type: ignore

# -----------------------------
# Adjustable Parameters
# -----------------------------
FER_CSV_PATH = os.path.join("dataset", "fer2013.csv")
BATCH_SIZE = 64  # Adjust to your system's capacity
EPOCHS = 2       # For testing. Increase as needed.
IMG_SIZE = (224, 224)  # (width, height). Reduce if memory is tight.
NUM_CLASSES = 7  # FER2013 typically has 7 classes (0..6)
LEARNING_RATE = 0.001

# If you still get MemoryErrors, reduce to e.g. (96, 96).
# This will significantly reduce memory usage.


class FERSequence(Sequence):
    """
    A custom Keras Sequence to load and preprocess FER2013 data
    on the fly, in batches, without storing everything in memory.
    """
    def __init__(self, df, batch_size, img_size, num_classes=7, shuffle=True):
        """
        :param df: A pandas DataFrame containing columns:
                   ['emotion', 'pixels', 'Usage']
        :param batch_size: Number of samples per batch
        :param img_size: Tuple (width, height) to resize images
        :param num_classes: Number of emotion classes for one-hot
        :param shuffle: Whether to shuffle indexes after each epoch
        """
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.img_width, self.img_height = img_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.df))
        self.on_epoch_end()  # shuffle at start if needed

    def __len__(self):
        """Number of batches in the Sequence per epoch."""
        return int(np.ceil(len(self.df) / float(self.batch_size)))

    def on_epoch_end(self):
        """Shuffle the dataset at the end of each epoch if desired."""
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, idx):
        """
        Generate one batch of data.
        """
        # 1) Compute which entries of DF this batch will contain
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, len(self.df))
        batch_indexes = self.indexes[start:end]

        # 2) Prepare arrays for images and labels
        X_batch = []
        y_batch = []

        for i in batch_indexes:
            row = self.df.iloc[i]
            emotion_label = row['emotion']
            pixel_str = row['pixels']
            pixel_values = list(map(int, pixel_str.split()))
            img_array = np.array(pixel_values).reshape(48, 48).astype('uint8')

            # Convert 48x48 grayscale -> desired size
            #  - Resize
            img_resized = cv2.resize(img_array, (self.img_width, self.img_height))
            #  - Convert to 3 channels
            img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
            #  - Normalize [0,1]
            img_norm = img_bgr / 255.0

            X_batch.append(img_norm)
            y_batch.append(emotion_label)

        X_batch = np.array(X_batch, dtype=np.float32)
        y_batch = to_categorical(y_batch, num_classes=self.num_classes)
        return X_batch, y_batch


def create_simple_cnn(input_shape, num_classes):
    """
    A small CNN for demonstration.
    You can replace this with your own model.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def main():
    # --------------------------------
    # 1. Load CSV
    # --------------------------------
    print("[INFO] Loading FER2013 CSV...")
    fer_data = pd.read_csv(FER_CSV_PATH)
    print("[INFO] Loaded DataFrame shape:", fer_data.shape)

    # --------------------------------
    # 2. Split by Usage
    # --------------------------------
    train_data = fer_data[fer_data['Usage'] == 'Training']
    val_data = fer_data[fer_data['Usage'] == 'PublicTest']
    test_data = fer_data[fer_data['Usage'] == 'PrivateTest']

    print("Train samples:", len(train_data))
    print("Val samples:", len(val_data))
    print("Test samples:", len(test_data))

    # --------------------------------
    # 3. Create Generators
    # --------------------------------
    print("[INFO] Creating data generators (Sequences)...")
    train_seq = FERSequence(train_data,
                            batch_size=BATCH_SIZE,
                            img_size=IMG_SIZE,
                            num_classes=NUM_CLASSES,
                            shuffle=True)

    val_seq = FERSequence(val_data,
                          batch_size=BATCH_SIZE,
                          img_size=IMG_SIZE,
                          num_classes=NUM_CLASSES,
                          shuffle=False)

    # Optionally, create a test_seq if you want to evaluate or do predictions:
    test_seq = FERSequence(test_data,
                           batch_size=BATCH_SIZE,
                           img_size=IMG_SIZE,
                           num_classes=NUM_CLASSES,
                           shuffle=False)

    # --------------------------------
    # 4. Build Model
    # --------------------------------
    print("[INFO] Building CNN model...")
    input_shape = (IMG_SIZE[1], IMG_SIZE[0], 3)  # (height, width, 3)
    model = create_simple_cnn(input_shape, NUM_CLASSES)

    # Compile
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # --------------------------------
    # 5. Train Model
    # --------------------------------
    print("[INFO] Training model with data generators...")
    model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=EPOCHS
    )

    # --------------------------------
    # 6. Optional: Evaluate on Test Set
    # --------------------------------
    print("[INFO] Evaluating on test set...")
    test_loss, test_acc = model.evaluate(test_seq)
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_acc)

    # --------------------------------
    # 7. Save Model
    # --------------------------------
    model.save("cnn_face_model.h5")
    print("[INFO] Model saved as cnn_face_model.h5")


if __name__ == "__main__":
    main()
