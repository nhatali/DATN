import flwr as fl
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import get_model
import sys

# ===== Device =====
DEVICE = torch.device("cpu")

# ===== Transform =====
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# ===== Lấy client id =====
if len(sys.argv) < 2:
    print("Usage: python client.py <client_id>")
    exit()

client_id = sys.argv[1]

# ===== Dataset =====
DATA_DIR = f"federated_data/client{client_id}"

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

print(f"Client {client_id} loaded {len(dataset)} images")

# ===== Model =====
model = get_model(len(dataset.classes)).to(DEVICE)

# ===== Train =====
def train():
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    for i, (images, labels) in enumerate(loader):
        if i > 20:   # 🔥 giảm thời gian chạy
            break

        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Client {client_id} finished training")

# ===== Evaluate =====
def evaluate():
    model.eval()
    correct = 0
    total = 0
    loss_total = 0

    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            if i > 10:  # giới hạn để chạy nhanh
                break

            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss_total += loss.item()

    acc = correct / total if total > 0 else 0
    avg_loss = loss_total / (i+1)

    print(f"Client {client_id} - Loss: {avg_loss:.4f}, Acc: {acc:.4f}")

    return avg_loss, acc

# ===== Flower Client =====
class Client(fl.client.NumPyClient):

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = dict(zip(model.state_dict().keys(), parameters))
        model.load_state_dict({k: torch.tensor(v) for k, v in state_dict.items()})

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train()
        return self.get_parameters(config), len(dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, acc = evaluate()
        return loss, len(dataset), {"accuracy": acc}

# ===== Start client =====
fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=Client()
)