import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
from PIL import Image
import sys

# ===== Load classes =====
with open("classes.txt", "r", encoding="utf-8") as f:
    classes = [line.strip() for line in f.readlines()]

# ===== Load model =====
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load("model_ucf.pth", map_location="cpu"))
model.eval()

# ===== Transform =====
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# ===== Đọc video =====
video_path = sys.argv[1]
cap = cv2.VideoCapture(video_path)

predictions = []

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Lấy mỗi 10 frame để tăng tốc
    if frame_count % 10 == 0:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = model(img)
            pred = torch.argmax(output, 1).item()
            predictions.append(pred)

        print(f"Frame {frame_count}: {classes[pred]}")

    frame_count += 1

cap.release()

# ===== Kết quả cuối =====
final_pred = max(set(predictions), key=predictions.count)
print("\nFinal Prediction:", classes[final_pred])