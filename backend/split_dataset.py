import os
import shutil
import random

SOURCE_DIR = r"C:\Users\Admin\Downloads\DATN\data\frames"
DEST_DIR = r"C:\Users\Admin\Downloads\DATN\federated_data"
NUM_CLIENTS = 3

random.seed(42)

classes = os.listdir(SOURCE_DIR)

for i in range(NUM_CLIENTS):
    for cls in classes:
        os.makedirs(os.path.join(DEST_DIR, f"client{i+1}", cls), exist_ok=True)

for cls in classes:
    cls_path = os.path.join(SOURCE_DIR, cls)
    images = os.listdir(cls_path)
    random.shuffle(images)

    chunks = [images[i::NUM_CLIENTS] for i in range(NUM_CLIENTS)]

    for i, chunk in enumerate(chunks):
        for img in chunk:
            src = os.path.join(cls_path, img)
            dst = os.path.join(DEST_DIR, f"client{i+1}", cls, img)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy(src, dst)

print("Dataset split completed.")