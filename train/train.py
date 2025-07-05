import sys


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train.py <start|train>")
        sys.exit(1)

    arg = sys.argv[1]
    if arg not in ["start", "train"]:
        print("Invalid argument. Use 'start' or 'train'.")
        sys.exit(1)


print("Starting TensorFlow and Library Setup...")

import numpy as np
import pandas as pd
import tensorflow as tf
import os
from tensorflow import keras
from tensorflow.keras.saving import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, Conv1D, MaxPooling2D, MaxPooling1D, BatchNormalization, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.applications import MobileNetV2
import tensorflow.keras.callbacks as callbacks
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import cv2
from glob import glob
import random
import gc
import dotenv
from pprint import pprint
from pathlib import Path
import collections

env_path = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), '.env'))
print(env_path)

dotenv.load_dotenv(dotenv_path=env_path)

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

    mem_limit = int(os.getenv('MEMORY_LIMIT'))  # Default to 10GB if not set
    if mem_limit <= 0:
        mem_limit = 1024
    print("VMemory Limit:", mem_limit, "MB")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                # tf.config.experimental.set_virtual_device_configuration(
                    # gpu,
                    # [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=mem_limit)]
                # )
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
# print("Current Working Directory:", os.getcwd())

image_size = (256, 256)
batch_size = 32
kernel_size = (3,3)
num_classes = len([i for i in Path(os.getenv('DATASET_PATH')).glob("*") if i.is_dir()])
val_train = 0.85


AUTOTUNE = tf.data.AUTOTUNE

def preprocess(image, label):
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
    return image, label

raw_train_ds = tf.keras.utils.image_dataset_from_directory(
    os.getenv('DATASET_PATH'),
    validation_split=1 - val_train,
    subset='training',
    seed=123,
    image_size=image_size,
    batch_size=batch_size,
    shuffle=True,
    label_mode='categorical',
)

train_ds = raw_train_ds.map(
    preprocess,
    num_parallel_calls=AUTOTUNE
).prefetch(buffer_size=AUTOTUNE)

raw_val_ds = tf.keras.utils.image_dataset_from_directory(
    os.getenv('DATASET_PATH'),
    validation_split=1 - val_train,
    subset='validation',
    seed=123,
    image_size=image_size,
    batch_size=batch_size,
    shuffle=True,
    label_mode='categorical',
)

val_ds = raw_val_ds.map(
    preprocess,
    num_parallel_calls=AUTOTUNE
).prefetch(buffer_size=AUTOTUNE)

train_gen = train_ds
val_gen = val_ds

class_names = raw_train_ds.class_names
print(type(class_names))
num_classes = len(class_names)
print("Class names:", class_names)

y_train_labels = []
for _, labels in train_gen:
    y_train_labels.extend(labels.numpy().argmax(axis=1))

y_train_labels = np.array(y_train_labels)

y_val_labels = []
for _, labels in val_gen:
    y_val_labels.extend(labels.numpy().argmax(axis=1))

y_val_labels = np.array(y_val_labels)



class_weight_dict = dict(zip(
    np.unique(y_train_labels),
    compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train_labels),
        y=y_train_labels
    )
))

counter_train = collections.Counter(y_train_labels)
counter_val = collections.Counter(y_val_labels)


print("Train sample distribution:")
#for class_index, count in counter_train.items():
#    print(f"Class {class_names[class_index]} ({class_index}): {count} images")

print("Validation sample distribution:")
#for class_index, count in counter_val.items():
#    print(f"Class {class_names[class_index]} ({class_index}): {count} images")



base_model = MobileNetV2(
    input_shape=(image_size[0], image_size[1], 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze the base model
#base_model.trainable = True
model = Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),  # Quan trọng

    keras.layers.Flatten(),

    keras.layers.Dense(256, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),

    keras.layers.Dense(128, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),

    keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

print(len(train_gen), len(val_gen))

epochs = 30

first_model = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    class_weight=class_weight_dict,
    verbose=1,
    callbacks=[
        callbacks.ModelCheckpoint(
            filepath=os.path.join(
                os.getenv("MODEL_PATH"),
                'food_epoch_{epoch:02d}.keras'
            ),
            save_weights_only=False,
            save_best_only=False, 
            monitor='val_loss'
        ),
        callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=5, 
            restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=3, 
            min_lr=1e-6, 
            verbose=1
        )
    ]
)
