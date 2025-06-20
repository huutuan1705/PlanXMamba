import os
from torch.utils.data import Dataset
from PIL import Image

class RiceDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_paths = [
            os.path.join(image_dir, fname)
            for fname in os.listdir(image_dir)
            if fname.lower().endswith(".jpg")
        ]
        self.transform = transform
        self.label_map = {
            '1': 0,
            '2': 1,
            '4': 2,
            '8': 3,
            '25': 4
        }

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        filename = os.path.basename(img_path)
        prefix = filename.split('_')[0]
        label = self.label_map.get(prefix, -1)

        if label == -1:
            raise ValueError(f"Unknown label prefix '{prefix}' in filename '{filename}'")

        if self.transform:
            image = self.transform(image)

        return image, label
