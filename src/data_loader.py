import os
import cv2
import numpy as np
from torch.utils.data import Dataset

class ChromosomeDataset(Dataset):
    """Assumes folder structure: data/raw/class_x/img1.png, etc.
    class_x is an integer representing chromosome type (1-24) or anomaly type.
    For simplicity, encode both chromosome type and anomaly into a single class index.
    """
    def init(self, root_dir, transform=None, img_size=(128,128)):
        self.img_paths = []
        self.labels = []
        for class_name in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            label = int(class_name)   # assuming folder name is integer label
            for fname in os.listdir(class_path):
                if fname.endswith(('.png', '.jpg', '.jpeg', '.tif')):
                    self.img_paths.append(os.path.join(class_path, fname))
                    self.labels.append(label)
        self.transform = transform
        self.img_size = img_size

    def len(self):
        return len(self.img_paths)

    def getitem(self, idx):
        img = cv2.imread(self.img_paths[idx], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, self.img_size)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)   # (1, H, W)
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label
