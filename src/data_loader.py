
import os
import cv2
import numpy as np
from torch.utils.data import Dataset

class ChromosomeDataset(Dataset):
    def init(self, root_dir, transform=None, img_size=(128,128)):
        self.paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(('.png','.jpg'))]
        self.transform = transform
        self.img_size = img_size

    def len(self):
        return len(self.paths)

    def getitem(self, idx):
        img = cv2.imread(self.paths[idx], cv2.IMREAD_GRAYSCALE)  # or RGB
        img = cv2.resize(img, self.img_size)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)  # shape (1, H, W)
        if self.transform:
            img = self.transform(img)
        return img.astype(np.float32)
