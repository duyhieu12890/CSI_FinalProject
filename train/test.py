import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import dotenv


env_path = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), '.env'))
print(env_path)
dotenv.load_dotenv()


print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)
print("Num CPUs Available: ", len(tf.config.list_physical_devices('CPU')))
print("Num CPUs Core(s): ", os.cpu_count())

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 1 = INFO, 2 = WARNING, 3 = ERROR

tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)
print("Num Inter Op Threads: ", tf.config.threading.get_inter_op_parallelism_threads())
print("Num Intra Op Threads: ", tf.config.threading.get_intra_op_parallelism_threads())

print(tf.config.list_physical_devices('GPU'))
gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))

if os.getenv('IS_CUDA') == 'True':

    mem_limit = int(os.getenv('MEMORY_LIMIT', "4096"))  # Default to 10GB if not set
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=mem_limit)]
                )
        except RuntimeError as e:
            # print(e)
            print("Error setting memory growth:", e)
            pass
else:
    print("CUDA is not enabled. Running on CPU only.")
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU

print(os.getcwd())


# print("Environment Variables:")
for key in dotenv.find_dotenv():
    value = os.getenv(key)
    if value is not None:
        print(f"{key}: {value}")
os.getenv('DATASET_PATH')