import os
import shutil
import json
from tqdm import tqdm
def load_category_file(filepath):
    labels = {}
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines[1:]:  # Bỏ dòng tiêu đề "id\tname"
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            index, label = parts
            try:
                index = int(index)
                label = label.strip().lower().replace(" ", "_")
                labels[index] = label
            except ValueError:
                continue
    return labels


def merge_labels(*dicts):
    label_set = set()
    for d in dicts:
        label_set.update(d.values())
    return {label: idx for idx, label in enumerate(sorted(label_set))}

def copy_and_rename_images(src_root, categories, label2id, out_root, metadata):
    for idx, label in tqdm(categories.items(), desc=f"📦 Đang xử lý {src_root}"):
        src_folder = os.path.join(src_root, str(idx))
        if not os.path.exists(src_folder):
            print(f"[!] Bỏ qua: {src_folder} không tồn tại")
            continue

        label_name = label
        dst_folder = os.path.join(out_root, label_name)
        os.makedirs(dst_folder, exist_ok=True)

        image_found = False
        for fname in os.listdir(src_folder):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            image_found = True
            src_img = os.path.join(src_folder, fname)
            new_name = f"{label_name}_{fname}"
            dst_img = os.path.join(dst_folder, new_name)
            shutil.copy(src_img, dst_img)
            metadata.append({
                "filename": os.path.relpath(dst_img, out_root),
                "label": label_name,
                "label_id": label2id[label_name]
            })

        if not image_found:
            print(f"[!] Không tìm thấy ảnh trong {src_folder}")


# Đường dẫn gốc
uec100_path = os.path.join(os.getcwd("WORKDATA_PATH"),"UECFOOD100")
uec256_path = os.path.join(os.getcwd("WORKDATA_PATH"),"UECFOOD256")
output_dataset = os.path.join(os.getcwd("DATASET_PATH"),"dataset_combined")
print(os.path.isfile(os.path.join(uec100_path, "category.txt")), os.path.isdir(uec256_path))

# Load label mappings từ category.txt
uec100_labels = load_category_file(os.path.join(uec100_path, "category.txt"))
uec256_labels = load_category_file(os.path.join(uec256_path, "category.txt"))

print(uec100_path, uec256_path)
# Gộp và chuẩn hóa nhãn
label2id = merge_labels(uec100_labels, uec256_labels)
id2label = {idx: label for label, idx in label2id.items()}
# Lưu labels.json
with open("labels.json", "w") as f:
    json.dump(id2label, f, indent=2)

# Tạo thư mục output + metadata
os.makedirs(output_dataset, exist_ok=True)
image_metadata = []

# Copy ảnh từ 100 & 256
copy_and_rename_images(os.path.join(uec100_path), uec100_labels, label2id, output_dataset, image_metadata)
copy_and_rename_images(os.path.join(uec256_path), uec256_labels, label2id, output_dataset, image_metadata)

# Lưu metadata ảnh
with open("image_data.json", "w") as f:
    json.dump(image_metadata, f, indent=2)

print(f"{len(image_metadata)} ảnh đã xử lý.")
