from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import gdown

# Define Google Drive File ID & Model Path
MODEL_ID = "1f42nicggpRV2hYhC9smtUuSPE6BfWDLJ"
MODEL_PATH = "cnn_face_model.h5"

# Download model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    print("[INFO] Model not found! Downloading from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={MODEL_ID}", MODEL_PATH, quiet=False)

# Load the trained model
print("[INFO] Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("[INFO] Model loaded successfully.")

# Define emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

app = FastAPI()

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Read file contents
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Debug: Show Original Image
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
        plt.title("Uploaded Image Before Processing")
        plt.show()

        # Preprocessing - Resize and Normalize
        img_resized = cv2.resize(img, (224, 224))  # Use your model's expected size
        img_normalized = img_resized / 255.0
        img_expanded = np.expand_dims(img_normalized, axis=0)

        # Debug: Show Processed Image
        plt.imshow(img_normalized)
        plt.title("Processed Image Before Model Prediction")
        plt.show()

        # Make prediction
        predictions = model.predict(img_expanded)
        predicted_index = np.argmax(predictions[0])
        predicted_emotion = emotion_labels[predicted_index]

        return {
            "predicted_emotion": predicted_emotion,
            "confidence_scores": predictions[0].tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
