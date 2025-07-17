import os
import tensorflow as tf
from tensorflow import keras
import dotenv
import io
from PIL import Image
import numpy as np
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as EfficientNetV2B2_preprocess_input


env_path = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), '.env'))
print(env_path)

dotenv.load_dotenv(dotenv_path=env_path)

print("TensorFlow version:", tf.__version__)
print("Keras version:", tf.keras.__version__)
print("Num CPUs Available: ", len(tf.config.list_physical_devices('CPU')))
print("Num CPUs Core(s): ", os.cpu_count())

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 1 = INFO, 2 = WARNING, 3 = ERROR

tf.config.threading.set_intra_op_parallelism_threads(os.cpu_count() // 2)
tf.config.threading.set_inter_op_parallelism_threads(os.cpu_count() // 2)
print("Num Inter Op Threads: ", tf.config.threading.get_inter_op_parallelism_threads())
print("Num Intra Op Threads: ", tf.config.threading.get_intra_op_parallelism_threads())

print(tf.config.list_physical_devices('GPU'))
gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))

if os.getenv('IS_CUDA') == 'True':

    mem_limit = int(os.getenv('MEMORY_LIMIT', "1840"))  # Default to 10GB if not set
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1536)]
                )
        except RuntimeError as e:
            # print(e)
            print("Error setting memory growth:", e)
            pass
else:
    print("CUDA is not enabled. Running on CPU only.")
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU

print(os.getcwd())


def model(path):
    return tf.keras.models.load_model(path)

def old_preprocess_image(file):
    img = Image.open(io.BytesIO(file)).convert("RGB")
    img = img.resize((256, 256))  # hoặc đúng theo kích thước model
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def preprocess_image(image):
    img = tf.image.decode_image(image, channels=3)
    img = tf.image.resize(img, [256, 256])
    img = tf.cast(img, tf.float32)

    img = EfficientNetV2B2_preprocess_input(img)

    img = tf.expand_dims(img, axis=0)
    return img