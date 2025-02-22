from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import numpy as np
import cv2
import os
import gdown # type: ignore
import tflite_runtime.interpreter as tflite # type: ignore

# Define Google Drive File ID & Model Path
MODEL_ID = "1pUtlrx6tp9rynR_w6LhPOiYme9BgC1WO"
MODEL_PATH = "cnn_face_model.tflite"

# Download model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    print("[INFO] Model not found! Downloading from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={MODEL_ID}", MODEL_PATH, quiet=False)

# Load the TensorFlow Lite model
print("[INFO] Loading TFLite model...")
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input & output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

app = FastAPI()

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Preprocess Image
        img_resized = cv2.resize(img, (224, 224))
        img_normalized = img_resized / 255.0
        img_expanded = np.expand_dims(img_normalized, axis=0).astype(np.float32)

        # Run inference using TFLite model
        interpreter.set_tensor(input_details[0]['index'], img_expanded)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])

        # Get prediction
        predicted_index = np.argmax(predictions[0])
        predicted_emotion = emotion_labels[predicted_index]

        return {"predicted_emotion": predicted_emotion, "confidence_scores": predictions[0].tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
