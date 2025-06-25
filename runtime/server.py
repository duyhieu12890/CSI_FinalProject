# app.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import tensorflow as tf
from PIL import Image
import numpy as np
import io

app = FastAPI()
model = tf.keras.models.load_model("model/your_model.h5")

def preprocess_image(file):
    img = Image.open(io.BytesIO(file)).convert("RGB")
    img = img.resize((256, 256))  # hoặc đúng theo kích thước model
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = preprocess_image(contents)
    preds = model.predict(img)
    return JSONResponse(content={"prediction": preds.tolist()})

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
