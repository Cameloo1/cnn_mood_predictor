from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import numpy as np
import cv2
import os
import gdown
import tflite_runtime.interpreter as tflite

# ✅ Correct Google Drive File ID (from your provided link)
MODEL_ID = "1pUtlrx6tp9rynR_w6LhPOiYme9BgC1WO"
MODEL_PATH = "cnn_face_model.tflite"

# Function to download the model
def download_model():
    print("[INFO] Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={MODEL_ID}", MODEL_PATH, quiet=False)

# ✅ Check if the model exists and is not corrupted
if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000:  # Ensure file isn't empty/corrupt
    print("[WARNING] Model file missing or corrupted. Downloading again...")
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)  # Remove corrupted file
    download_model()

# ✅ Load TensorFlow Lite model
try:
    print("[INFO] Loading TFLite model...")
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    print("[INFO] Model loaded successfully!")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    raise HTTPException(status_code=500, detail="Model loading failed. Check the model file.")

# Define emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

app = FastAPI()

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        # ✅ Read and decode the uploaded image
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # ✅ Preprocess Image (Resize & Normalize)
        img_resized = cv2.resize(img, (224, 224))
        img_normalized = img_resized / 255.0
        img_expanded = np.expand_dims(img_normalized, axis=0).astype(np.float32)

        # ✅ Run inference using TFLite model
        interpreter.set_tensor(interpreter.get_input_details()[0]['index'], img_expanded)
        interpreter.invoke()
        predictions = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

        # ✅ Get prediction
        predicted_index = np.argmax(predictions[0])
        predicted_emotion = emotion_labels[predicted_index]

        return {"predicted_emotion": predicted_emotion, "confidence_scores": predictions[0].tolist()}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
