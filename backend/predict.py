import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Đọc class từ file
with open("classes.txt", "r", encoding="utf-8") as f:
    classes = [line.strip() for line in f.readlines()]

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load("model_ucf.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

img = Image.open("test.jpg").convert("RGB")
img = transform(img).unsqueeze(0)

with torch.no_grad():
    output = model(img)
    pred = torch.argmax(output, 1)

print("Prediction:", classes[pred.item()])