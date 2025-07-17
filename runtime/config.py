import os
import dotenv


env_path = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), '.env'))
print(env_path)

dotenv.load_dotenv(dotenv_path=env_path)

index_model = ["uecfood256", "uecfood100"]
model = {
    "uecfood256": {
        "name": "UECFOOD256",
        "Description": " ",
        "metadata": "metadata/uecfoos256.json",
        "model": os.path.join(os.getenv("MODEL_PATH"), "uecfood256.keras")
    },
    "uecfood100": {
        "name" : "UECFOOD100",
        "Description": "",
        "metadata": "metadata/uecfood100.json",
        "model": os.path.join(os.getenv("MODEL_PATH"), "uecfood100.keras")
    }
}

LIMIT_TOP_K = 10
