from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import numpy as np
import cv2
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("cnn_face_model.h5")

# Define emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

app = FastAPI()

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # Read file contents
    contents = await file.read()
    
    # Convert image to NumPy array
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    # 🔴 Check if the image was loaded correctly
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image. Could not be read by OpenCV.")
    
    # Resize to model's input size
    img_resized = cv2.resize(img, (224, 224))  # Change size if needed
    img_normalized = img_resized / 255.0  # Normalize pixel values [0,1]
    img_expanded = np.expand_dims(img_normalized, axis=0)  # Expand for batch
    
    # Make prediction
    predictions = model.predict(img_expanded)
    
    # Get predicted class
    predicted_index = np.argmax(predictions[0])
    predicted_emotion = emotion_labels[predicted_index]
    
    return {
        "predicted_emotion": predicted_emotion,
        "confidence_scores": predictions[0].tolist()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
