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
from tensorflow.keras.saving import load_model # pyright: ignore[reportMissingImports]
from tensorflow.keras.preprocessing import image # pyright: ignore[reportMissingImports] # pyright: ignore[reportMissingImports]
from tensorflow.keras.preprocessing.image import ImageDataGenerator # pyright: ignore[reportMissingImports]
from tensorflow.keras.preprocessing.text import Tokenizer # pyright: ignore[reportMissingImports]
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, Conv1D, MaxPooling2D, MaxPooling1D, BatchNormalization, Embedding # pyright: ignore[reportMissingImports]
from tensorflow.keras.models import Sequential # pyright: ignore[reportMissingImports]
from tensorflow.keras.utils import pad_sequences # pyright: ignore[reportMissingImports]
from tensorflow.keras.applications import MobileNetV2, EfficientNetV2B2 # pyright: ignore[reportMissingImports]
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as EfficientNetV2B2_preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint # pyright: ignore[reportMissingImports]
import tensorflow.keras.callbacks as callbacks # pyright: ignore[reportMissingImports]
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
import json
import config
import PIL.Image as PILImage

env_path = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), '.env'))
print(env_path)

dotenv.load_dotenv(dotenv_path=env_path)

print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)
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

    mem_limit = int(os.getenv('MEMORY_LIMIT', 10240))  # Default to 10GB if not set
    if mem_limit <= 0:
        mem_limit = 1024
    print("VMemory Limit:", mem_limit, "MB")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                # tf.config.experimental.set_virtual_device_configuration(
                #     gpu,
                #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=mem_limit)]
                # )
        except RuntimeError as e:
            print("Error setting memory growth:", e)
            pass
else:
    print("CUDA is not enabled or not available. Running on CPU only.")
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU

print(os.getcwd())


# print("Environment Variables:")
for key in dotenv.find_dotenv():
    value = os.getenv(key)
    if value is not None:
        print(f"{key}: {value}")
os.getenv('DATASET_PATH')
# print("Current Working Directory:", os.getcwd())

data_root_dir = os.path.join(os.getenv("DATASET_PATH"), "UECFOOD256")
AUTOTUNE = tf.data.AUTOTUNE
print(data_root_dir)

id_to_name = {}
name_to_id = {}
with open(os.path.join(data_root_dir, "category.txt")) as file:
    next(file)
    for line in file:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            food_id = int(parts[0])
            food_name = parts[1]
            id_to_name[food_id] = food_name
            name_to_id[food_name] = food_id

sorted_food_ids = sorted(id_to_name.keys())
class_names = [id_to_name[food_id] for food_id in sorted_food_ids]

label_to_index = {name: i for i, name in enumerate(class_names)}
index_to_label = {i: name for i, name in enumerate(class_names)}


all_image_paths = []
all_image_labels_raw = []

valid_image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')

print("skipping not valid image:", end=" ")

# Duyệt qua các thư mục con theo ID (nếu bạn tổ chức theo ID)
for food_id in sorted_food_ids:
    class_dir = os.path.join(data_root_dir, str(food_id)) # Thư mục có tên là ID số
    if os.path.isdir(class_dir):
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(valid_image_extensions):
                img_path = os.path.join(class_dir, img_name)
                all_image_paths.append(img_path)
                # Chuyển ID số sang chỉ mục tương ứng trong class_names đã sắp xếp
                all_image_labels_raw.append(label_to_index[id_to_name[food_id]])
            else:
                print(os.path.join(str(food_id), img_name), end=" ")



num_classes = len(class_names)
all_image_labels_one_hot = tf.keras.utils.to_categorical(all_image_labels_raw, num_classes=num_classes)

# Chia dữ liệu thành tập huấn luyện, xác thực và kiểm tra
train_paths, test_paths, train_labels, test_labels = train_test_split(
    all_image_paths, all_image_labels_one_hot, test_size=0.2, random_state=42, stratify=all_image_labels_raw # stratify theo raw labels
)
train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_paths, train_labels, test_size=0.25, random_state=42, stratify=[np.argmax(label) for label in train_labels] # stratify theo raw labels của tập đã chia
)

print(f"Tổng số ảnh: {len(all_image_paths)}")
print(f"Số ảnh huấn luyện: {len(train_paths)}")
print(f"Số ảnh xác thực: {len(val_paths)}")
print(f"Số ảnh kiểm tra: {len(test_paths)}")
print(f"Số lớp: {num_classes}")

def read_bb_info(class_dir):
    bb_path = os.path.join(class_dir, "bb_info.txt")
    if not os.path.exists(bb_path):
        return {}
    
    with open(bb_path, 'r') as file:
        lines = file.readlines()

    # Lọc bỏ dòng header (dòng đầu có chứa chữ 'img')
    lines = [line for line in lines if not line.startswith("img")]

    # Viết tạm file cleaned
    tmp_path = bb_path + ".cleaned"
    with open(tmp_path, 'w') as f:
        f.writelines(lines)

    df = pd.read_csv(tmp_path, sep='\s+', header=None, names=['img', 'x1', 'y1', 'x2', 'y2'])
    bb_dict = dict()
    for row in df.itertuples(index=False):
        # print(row.x1, " ", row.y1, " ", row.x2, " ", row.y2)

        bb_dict[row.img] = [int(row.x1), int(row.y1), int(row.x2), int(row.y2)]
    os.remove(tmp_path)
    return bb_dict

total_bb_info = {}


print("Đang xử lý bounding box... ", end="", flush=True)
for food_id in sorted_food_ids:
    class_dir = os.path.join(data_root_dir, str(food_id))
    if os.path.isdir(class_dir):
        bb_dict = read_bb_info(class_dir)
        # print(bb_dict)
        for filename, bb in bb_dict.items():
            total_bb_info[os.path.join(str(class_dir), str(filename))] = bb
    print(food_id, end=" ", flush=True)

print(f"\nĐã xử lý Bounding box cho {len(total_bb_info)} ảnh")



def decode_and_crop(img_raw, bb):
    img = tf.io.decode_jpeg(img_raw, channels=3)

    img_shape = tf.shape(img)
    h = img_shape[0]
    w = img_shape[1]

    x1 = tf.clip_by_value(bb[0],0,w-1)
    y1 = tf.clip_by_value(bb[1],0,h-1)
    x2 = tf.clip_by_value(bb[2],x1 + 1,w)
    y2 = tf.clip_by_value(bb[3],y1 + 1,h)

    crop_img = tf.image.crop_to_bounding_box(img, offset_height=y1, offset_width=x1, target_height=y2 - y1, target_width=x2 - x1)
    resized_img = tf.image.resize(crop_img, [config.IMAGE_SIZE[0], config.IMAGE_SIZE[1]])
    resized_img = tf.cast(resized_img, tf.float32) / 255.0
    return resized_img

# --- 3. Hàm tải và tiền xử lý ảnh ---
def decode_img(img_raw): # Đổi tên biến đầu vào cho rõ ràng
    # Thay thế tf.image.decode_jpeg(img_raw, channels=3)
    img = tf.io.decode_jpeg(img_raw, channels=3) # Sử dụng tf.io.decode_image
    # Kiểm tra kích thước của ảnh sau khi giải mã để đảm bảo nó hợp lệ
    # Điều này giúp loại bỏ các file không phải ảnh nhưng có thể được decode (ví dụ: file rỗng)


    img = tf.image.resize(img, [config.IMAGE_SIZE[0], config.IMAGE_SIZE[1]])
    img = tf.cast(img, tf.float32) / 255.0
    return img

# def process_path(file_path, label):
#     # Sử dụng tf.io.read_file và tf.py_function để xử lý lỗi tốt hơn nếu cần
#     # Hoặc giữ nguyên và thêm try-except trong decode_img nếu lỗi xảy ra trong map
#     img_raw = tf.io.read_file(file_path)
#     # Bao bọc decode_img trong tf.py_function để bắt lỗi Python và bỏ qua ảnh lỗi
#     # Điều này giúp pipeline không bị crash nhưng bạn cần xử lý đầu ra
#     try:
#         img = decode_img(img_raw)
#         return img, label
#     except tf.errors.InvalidArgumentError as e:
#         tf.print(f"Skipping corrupt or invalid image: {file_path} - {e}")
#         # Trả về một giá trị đặc biệt hoặc bỏ qua phần tử này (phức tạp hơn trong tf.data)
#         # Cách đơn giản nhất để tiếp tục là trả về một tensor rỗng và lọc sau
#         return tf.zeros([config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], 3], dtype=tf.float32), tf.zeros([num_classes], dtype=tf.float32)

def process_path(file_path, label):
    def py_process(fp_str):
            fp_str = fp_str.numpy().decode("utf-8")

            # Đọc ảnh gốc
            img_tensor = tf.io.read_file(fp_str)
            img_decoded = tf.io.decode_jpeg(img_tensor, channels=3)
            shape = img_decoded.shape

            # Lấy bounding box
            if fp_str in total_bb_info:
                bb = total_bb_info[fp_str]
                x1, y1, x2, y2 = bb
            else:
                x1, y1 = 0, 0
                x2, y2 = shape[1], shape[0]  # width, height

            # Crop ảnh
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(shape[1], x2)
            y2 = min(shape[0], y2)

            crop_img = img_decoded[y1:y2, x1:x2, :]
            resized_img = cv2.resize(crop_img.numpy(), (config.IMAGE_SIZE[1], config.IMAGE_SIZE[0]))

            return resized_img.astype(np.float32)

    # Dùng tf.py_function để wrap function Python
    img = tf.py_function(
        func=py_process,
        inp=[file_path],
        Tout=tf.float32
    )
    img.set_shape([config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], 3])
    img = EfficientNetV2B2_preprocess_input(img)
    return img, label

# Trong prepare_dataset, sau map(process_path), bạn có thể lọc các phần tử rỗng:
# dataset = dataset.filter(lambda img, label: tf.reduce_sum(img) > 0)


def augment_data(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.5)
    img = tf.image.random_contrast(img, 0.7, 1.3)
    img = tf.image.random_saturation(img, 0.7, 1.3)
    img = tf.image.random_hue(img, 0.1)
    return img, label

# --- 4. Xây dựng tf.data.Dataset Pipeline ---
def prepare_dataset(image_paths, image_labels, is_training=True, cache_name=False):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, image_labels))

    # Áp dụng hàm xử lý đường dẫn để tải và tiền xử lý ảnh
    dataset = dataset.map(process_path, num_parallel_calls=AUTOTUNE)

    if bool(os.getenv("USE_DATASET_CACHE")) and cache_name != False:
        print("Dataset cache enable on ", cache_name, " Directory will use:",os.path.join(os.getenv("DATASET_CACHE_PATH"), cache_name))
        os.makedirs(os.path.join(os.getenv("DATASET_CACHE_PATH"), cache_name), exist_ok=True)
        dataset.cache(os.path.join(os.getenv("DATASET_CACHE_PATH"), cache_name))

    if is_training:
        dataset = dataset.shuffle(buffer_size=128) # Đảo trộn dữ liệu huấn luyện
        dataset = dataset.map(augment_data, num_parallel_calls=AUTOTUNE) # Áp dụng tăng cường dữ liệu

    # Nhóm các ảnh lại thành batch
    dataset = dataset.batch(config.BATCH_SIZE)


    # Nạp trước các batch để GPU không phải chờ dữ liệu
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    return dataset



# --- Xây dựng tf.data.Dataset Pipeline ---
train_ds = prepare_dataset(train_paths, train_labels, is_training=True, cache_name="train")
val_ds = prepare_dataset(val_paths, val_labels, is_training=False, cache_name="val")
test_ds = prepare_dataset(test_paths, test_labels, is_training=False, cache_name="test")

train_int_labels = [np.argmax(label) for label in train_labels]

y_int_labels = [np.argmax(i) for i in all_image_labels_one_hot]
unique_classes = np.unique(train_int_labels)
class_weights = compute_class_weight(class_weight="balanced",classes=unique_classes, y=train_int_labels)
class_weights_dict = dict(zip(unique_classes, class_weights))

 
# --- 5. Lựa Chọn Mô Hình (Transfer Learning với ResNet50) ---
# Tải mô hình ResNet50 đã được huấn luyện trước trên ImageNet, bỏ qua lớp đầu ra
base_model = tf.keras.applications.EfficientNetV2B2(
    weights='imagenet',
    include_top=False,
    input_shape=(config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], 3)
)

# Đóng băng các lớp của base_model để không huấn luyện lại chúng
base_model.trainable = False

# Thêm các lớp mới cho bài toán phân loại của bạn
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x) # Lớp gộp trung bình toàn cục
x = tf.keras.layers.Dense(1024, activation='relu')(x) # Lớp ẩn
predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x) # Lớp đầu ra

model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

# --- 6. Huấn Luyện Mô Hình ---
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


early_stop = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, min_lr=1e-6)
checkpoint = ModelCheckpoint("best_model_epoch_{epoch}.keras", monitor='val_accuracy', save_best_only=True)

history_head = model.fit(
    train_ds,
    epochs=20,
    validation_data = val_ds,
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1,
    class_weight=class_weights_dict
)


model.summary()

print("Model head trainned, fine tunning...")

for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=[early_stop, reduce_lr, checkpoint],
    class_weight=class_weights_dict,
    verbose=1
)



# --- 7. Đánh Giá Mô Hình ---
print("\nĐánh giá trên tập kiểm tra:")
loss, accuracy = model.evaluate(test_ds, verbose=1)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# --- 8. Lưu Mô Hình (Sau khi hài lòng với kết quả) ---
model.save(os.path.join(os.getenv("MODEL_PATH"), "UECFOOD256.keras"))
