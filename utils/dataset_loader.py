import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class ActionDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        self.transform = transforms.ToTensor()

        self.classes = os.listdir(root_dir)

        for label, action in enumerate(self.classes):
            action_path = os.path.join(root_dir, action)

            for subfolder in os.listdir(action_path):
                sub_path = os.path.join(action_path, subfolder)

                if not os.path.isdir(sub_path):
                    continue

                for img in os.listdir(sub_path):
                    if img.endswith(".jpg"):
                        self.samples.append(
                            (os.path.join(sub_path, img), label)
                        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label