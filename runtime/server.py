from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import threading
import subprocess
import psutil
import libbase
import platform

from firebase_admin import db
libbase.load_firebase()

def open_tunnel_and_push_url_to_firebase():
    print("Starting Cloudflare Tunnel...")
    proc = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", "http://127.0.0.1:8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    found_url = ""
    for line in proc.stdout:
        if "https://" in line and ".trycloudflare.com":
            found_line = line.strip()
            if "https://" in found_line and ".trycloudflare.com":
                found_url = found_line
                print("Finded url:", found_url)
                break
    ref = libbase.get_root_db()
    ref.child("host").child("model").child(platform.node()).set(found_url)

threading.Thread(target=open_tunnel_and_push_url_to_firebase, daemon=True).start()

print("Starting Tensorflow and FastAPI server...")
import tensorflow as tf
import os
import dotenv
import Model
# import module.process_string as process_string
import json
import config

app = FastAPI()

env_path = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), '.env'))
print(env_path)

dotenv.load_dotenv(dotenv_path=env_path)

model = Model.model(os.path.join(os.getenv("MODEL_PATH"), "uecfood100.keras"))

@app.get("/")
async def root():
    return {"message": "Welcome to the Food Image Classification API. Use POST /predict to classify images."}

@app.get("/isalive")
async def isAlive():
    return True

@app.get("/host_info")
def get_basic_host_info():
    return JSONResponse(content={ 
        "system": platform.system(),
        "node": platform.node(),
        "release": platform.release(),
        "machine": platform.machine(),
        "tensorflow_version": tf.__version__,
        "python_version": platform.python_version(),
        "system_memory_usable": f"{psutil.virtual_memory().total / (1024**3):.2f}",
    })

models = {}

def load_model():
    global models
    for name in config.index_model:
        path = config.model[name]["model"]
        if os.path.exists(path):
            models[name] = Model.model(path)
            print(f"✅ Loaded model: {name}")
        else:
            print(f"❗ Model not Found: {name}")

@app.get("/available_models")
async def get_available_models():
    return JSONResponse(content=config.model)

@app.get("{id}/metadata")
async def metadata(id:str):
    file = open(os.path.join("metadata", config.model[id]["metadata"]), 'rb')
    content = file.read()
    file.close()
    return JSONResponse(content=json.loads(content.decode('utf-8')))


def get_food_info(index:int, metadata:list):
    metadata_index = index + 1
    for item in metadata:
        if item["index"] == metadata_index:
            return item
    return {"name": "Unknown", "id": "Unknown", "description": "No description available."}


@app.post("/{id}/predict")
async def predict(id:str, file: UploadFile = File(...), top_k: int = 10):
    if id not in config.index_model:
        return JSONResponse(status_code=404, content={"error": "Model not found"})
    
    contents = await file.read()
    img = Model.preprocess_image(contents)
    preds = models[id].predict(img)[0]

    metadata = json.loads(open(config[id]["metadata"]), 'r', encoding="utf-8").read()["data"]
    
    pred_tensor = tf.convert_to_tensor(preds)
    
    top_indices = tf.argsort(pred_tensor, direction='DESCENDING')[:top_k].numpy()

    results = []
    
    for index in top_indices:
        score = float(pred_tensor[index]) * 100
        food_info = get_food_info(index, metadata)
        results.append({
            "name": food_info["name"],
            "id": food_info["id"],
            "index": int(index),
            "percentage": score,
            "description": food_info.get("description", "No description available.")
        })
    print(results)
    # return JSONResponse(content={"prediction": preds.tolist()})
    return JSONResponse(content={"results": results})


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)