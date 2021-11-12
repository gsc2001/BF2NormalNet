from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import torch


class CustomDataset(Dataset):
    def __init__(self, img_dir: str, scale: float = 0.5, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.scale = scale

        self.ids = [name for name in sorted(os.listdir(img_dir)) if not name.startswith('.')]

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, "Scale is too small"
        pil_img = pil_img.resize((newW, newH), Image.NEAREST)

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))

        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, index):
        idx = self.ids[index]
        rgb_file = os.path.join(self.img_dir, idx)
        rgb = Image.open(rgb_file)

        rgb = self.preprocess(rgb, self.scale)

        # if self.transform:
        #     rgb = self.transform(rgb)

        return {
            "rgb": torch.from_numpy(rgb).float()
        }
