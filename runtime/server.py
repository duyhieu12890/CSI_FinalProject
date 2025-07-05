# app.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import tensorflow as tf
import io
import os
import dotenv
import Model

app = FastAPI()

env_path = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), '.env'))
print(env_path)

dotenv.load_dotenv(dotenv_path=env_path)

model = Model.model(os.path.join(os.getenv("MODEL_PATH", "food.keras")))


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Model.preprocess_image(contents)
    preds = model.predict(img)
    return JSONResponse(content={"prediction": preds.tolist()})

@app.get("/")
async def root():
    return {"message": "Welcome to the Food Image Classification API. Use POST /predict to classify images."}

@app.get("/status")
async def status():
    return {
        "tensorflow_version": tf.__version__,
        "python_version": os.sys.version.split()[0],
        "os_info": os.uname().sysname + " " + os.uname().release,
        "system_memory": f"{os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024. ** 3):.2f} GB",
        "cores": os.cpu_count()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
