import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from PIL import Image as PILImage # Dùng Pillow để mở và xử lý ảnh
import config
import dotenv


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





# --- 1. Tải mô hình đã huấn luyện ---
# Sử dụng @tf.function để tối ưu hóa hiệu suất dự đoán (nếu dùng TensorFlow)

@tf.function # Bỏ comment nếu muốn dùng @tf.function
def load_food_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        QMessageBox.critical(None, "Lỗi", f"Không thể tải mô hình: {e}")
        return None

# --- 2. Tải và xử lý file category.txt ---
def load_class_names(category_file_path):
    id_to_name = {}
    try:
        with open(category_file_path, 'r', encoding='utf-8') as f:
            next(f) # Bỏ qua dòng tiêu đề
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    food_id = int(parts[0])
                    food_name = parts[1]
                    id_to_name[food_id] = food_name
        
        # Sắp xếp tên lớp theo ID để đảm bảo thứ tự nhất quán với mô hình
        sorted_food_ids = sorted(id_to_name.keys())
        class_names = [id_to_name[food_id] for food_id in sorted_food_ids]
        index_to_label = {i: name for i, name in enumerate(class_names)}
        return index_to_label
    except Exception as e:
        QMessageBox.critical(None, "Lỗi", f"Không thể đọc file category.txt: {e}")
        return {}

# --- Cấu hình các tham số cho mô hình ---
IMG_HEIGHT = config.IMAGE_SIZE[0]
IMG_WIDTH = config.IMAGE_SIZE[1]
MODEL_PATH = os.path.join(os.getenv("MODEL_PATH"), "uecfood256.keras")
CATEGORY_FILE_PATH = os.path.join(os.getenv("DATASET_PATH"), "UECFOOD256", "category.txt")

# --- Lớp ứng dụng GUI chính ---
class FoodRecognitionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nhận Diện Món Ăn (PyQt)")
        self.setGeometry(100, 100, 800, 600) # (x, y, width, height)

        self.model = None
        self.index_to_label = {}
        self.image_path = None

        self.init_ui()
        self.load_resources()

    def init_ui(self):
        # Tạo layout chính (Vertical)
        main_layout = QVBoxLayout()

        # Tiêu đề
        title_label = QLabel("Ứng dụng Nhận Diện Món Ăn")
        title_label.setFont(tf.keras.layers.experimental.preprocessing.TextVectorization.make_spec('Arial', 20)) # Kích thước font
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)

        # Khu vực hiển thị ảnh
        self.image_label = QLabel("Ảnh sẽ hiển thị ở đây")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(400, 400) # Kích thước cố định cho ảnh
        self.image_label.setStyleSheet("border: 1px solid gray;") # Viền cho dễ nhìn
        main_layout.addWidget(self.image_label, alignment=Qt.AlignCenter)

        # Khu vực nút bấm (Horizontal)
        button_layout = QHBoxLayout()
        self.load_button = QPushButton("Tải Ảnh Lên")
        self.load_button.clicked.connect(self.load_image)
        button_layout.addWidget(self.load_button)

        self.predict_button = QPushButton("Dự Đoán")
        self.predict_button.clicked.connect(self.predict_image)
        self.predict_button.setEnabled(False) # Ban đầu vô hiệu hóa nút dự đoán
        button_layout.addWidget(self.predict_button)
        
        main_layout.addLayout(button_layout)

        # Khu vực hiển thị kết quả
        self.result_label = QLabel("Kết quả dự đoán: Chưa có")
        self.result_label.setFont(tf.keras.layers.experimental.preprocessing.TextVectorization.make_spec('Arial', 14))
        self.result_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.result_label)

        self.set_layout(main_layout)

    def load_resources(self):
        # Tải mô hình và tên lớp khi khởi động ứng dụng
        self.model = load_food_model(MODEL_PATH)
        if self.model:
            self.index_to_label = load_class_names(CATEGORY_FILE_PATH)
            if not self.index_to_label: # Nếu không tải được tên lớp, vô hiệu hóa dự đoán
                self.predict_button.setEnabled(False)
                QMessageBox.warning(self, "Cảnh báo", "Không thể tải tên lớp. Chức năng dự đoán sẽ bị vô hiệu hóa.")
            else:
                QMessageBox.information(self, "Thông báo", "Mô hình và dữ liệu đã sẵn sàng!")
        else:
            QMessageBox.critical(self, "Lỗi", "Không thể khởi tạo ứng dụng do lỗi tải mô hình.")
            # Đóng ứng dụng nếu mô hình không tải được
            QApplication.quit()


    def load_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Chọn tệp ảnh", "", 
                                                   "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)", options=options)
        if file_name:
            self.image_path = file_name
            # Hiển thị ảnh trên GUI
            pixmap = QPixmap(self.image_path)
            pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(pixmap)
            self.image_label.setText("") # Xóa text placeholder

            # Kích hoạt nút dự đoán khi ảnh đã được tải
            self.predict_button.setEnabled(True)
            self.result_label.setText("Kết quả dự đoán: Đã sẵn sàng để dự đoán")

    def preprocess_image(self, image_path):
        # Dùng PIL để mở ảnh vì nó linh hoạt hơn cho các định dạng khác nhau
        img = PILImage.open(image_path).convert('RGB')
        img_array = np.array(img)
        # Resize ảnh về kích thước mô hình yêu cầu
        img_array = tf.image.resize(img_array, [IMG_HEIGHT, IMG_WIDTH])
        # Chuẩn hóa pixel về 0-1
        img_array = img_array / 255.0
        # Thêm chiều batch (1 ảnh)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict_image(self):
        if not self.image_path:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng tải lên một bức ảnh trước.")
            return
        if not self.model:
            QMessageBox.critical(self, "Lỗi", "Mô hình chưa được tải hoặc bị lỗi.")
            return
        if not self.index_to_label:
            QMessageBox.critical(self, "Lỗi", "Dữ liệu tên lớp chưa được tải.")
            return

        try:
            # Tiền xử lý ảnh
            processed_image = self.preprocess_image(self.image_path)

            # Dự đoán
            predictions = self.model.predict(processed_image)
            predicted_class_index = np.argmax(predictions)
            predicted_class_name = self.index_to_label.get(predicted_class_index, "Không xác định")
            confidence = predictions[0][predicted_class_index] * 100

            # Hiển thị kết quả
            result_text = f"Đây có vẻ là: <b>{predicted_class_name}</b><br>" \
                          f"Độ tin cậy: <span style='color:blue;'>{confidence:.2f}%</span>"
            self.result_label.setText(result_text)

            # Hiển thị Top 5 dự đoán
            top_5_indices = np.argsort(predictions[0])[-5:][::-1]
            top_5_results = "<br>---<br>Top 5 dự đoán:<br>"
            for i, idx in enumerate(top_5_indices):
                food_name = self.index_to_label.get(idx, "Không xác định")
                conf = predictions[0][idx] * 100
                top_5_results += f"{i+1}. {food_name}: {conf:.2f}%<br>"
            self.result_label.setText(result_text + top_5_results)

        except Exception as e:
            QMessageBox.critical(self, "Lỗi Dự đoán", f"Đã xảy ra lỗi trong quá trình dự đoán: {e}")


# --- Chạy ứng dụng ---
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FoodRecognitionApp()
    window.show()
    sys.exit(app.exec_())