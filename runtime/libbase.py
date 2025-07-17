import firebase_admin
from firebase_admin import credentials, db
import json
import pyrebase
import os
import dotenv

env_path = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), '.env'))
print(env_path)

dotenv.load_dotenv(dotenv_path=env_path)


def load_firebase():
    print("Loading Firebase...")
    
    # Parse chuỗi JSON từ secrets thành dict
    try:
        firebase_creds = json.loads(os.getenv("FIREBASE_SECRETS_STRING"))

        # Tạo credentials từ dict
        cred = firebase_admin.credentials.Certificate(firebase_creds)

        # Khởi tạo Firebase
        firebase_admin.initialize_app(cred, {
            'databaseURL': os.getenv("DATABASE_URL")
        })
    except ValueError as e:
        print(f"Error parsing Firebase secrets or had initialized app: {e}")
    except Exception as e:
        print(f"Error initializing Firebase: {e}")

def auth():
    return pyrebase.initialize_app(json.loads(os.getenv("FIREBASE_CONFIG_STRING"))).auth()


def get_root_db():
    IS_WORK = True
    try:
        load_firebase()
    except Exception as e:
        IS_WORK = False
        print(f"Error getting root database reference: {e}")
        return None
    if IS_WORK:
        print(db.reference("/"))
        return db.reference("/")
    else: return {}


def get_spec_db(directory: str):    
    IS_WORK = True
    try:
        load_firebase()
    except Exception as e:
        IS_WORK = False
        print(f"Error getting root database reference: {e}")
        return None
    if IS_WORK:
        print(db.reference(directory))
        return db.reference(directory)
    else: return {}
