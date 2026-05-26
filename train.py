import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ====== Cấu hình ======
DATA_DIR = r"C:\Users\Admin\Downloads\DATN\data\frames"
BATCH_SIZE = 16
EPOCHS = 10
MODEL_PATH = "model_ucf.pth"

# ====== Transform ảnh ======
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ====== Load dataset ======
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

print("Classes:", dataset.classes)
print("Total images:", len(dataset))

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ====== Tạo model ======
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))

# ====== Loss + Optimizer ======
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ====== Training ======
for epoch in range(EPOCHS):

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(loader)
    accuracy = 100 * correct / total

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.2f}%")

# ====== Lưu model ======
torch.save(model.state_dict(), MODEL_PATH)
print("Training Finished")
print(f"Model saved to {MODEL_PATH}")

# ====== Lưu danh sách class để predict dùng lại ======
with open("classes.txt", "w", encoding="utf-8") as f:
    for c in dataset.classes:
        f.write(c + "\n")

print("Classes saved to classes.txt")