print("Starting Tensorflow and FastAPI server...")
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import tensorflow as tf
import io
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
 
@app.get("/info")
async def status():
    return {
        "tensorflowzversion": tf.__version__,
        "python_version": os.sys.version.split()[0],
        "os_info": os.uname().sysname + " " + os.uname().release,
        "system_memory": f"{os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024. ** 3):.2f} GB",
        "cores": os.cpu_count()
    }


@app.get("/metadata")
async def metadata():
    file = open(os.path.join("metadata", "uecfood100.json"), 'rb')
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
async def predict(file: UploadFile = File(...), top_k: int = 10):
    contents = await file.read()
    img = Model.preprocess_image(contents)
    preds = model.predict(img)[0]

    metadata = json.loads(open(os.path.join("metadata", "uecfood100.json"), 'r', encoding="utf-8").read())["data"]
    
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

@app.post("/storage/upload")
async def upload(file: UploadFile= File(...)):
    pass


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
