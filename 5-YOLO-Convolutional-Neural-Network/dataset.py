import json
import os

import numpy as np
import torch
from PIL import Image


class coco_dataset(torch.utils.data.Dataset):
    def __init__(self, images_path="train2017", image_size=416,
                 labels_path="annotations/instances_train2017.json"):
        super().__init__()
        # Resolve to absolute paths so spawned DataLoader workers don't depend on cwd.
        self.images_path = os.path.abspath(images_path)
        self.labels_path = os.path.abspath(labels_path)
        self.image_size = image_size

        with open(self.labels_path) as f:
            coco = json.load(f)

        self.cat_to_idx = {cat['id']: idx for idx, cat in enumerate(coco['categories'])}

        self.labels = {}
        for label in coco["annotations"]:
            self.labels.setdefault(label["image_id"], []).append(label)

        self.images = coco["images"]

    def __len__(self):
        return len(self.images)

    def _load(self, index):
        img_info = self.images[index]
        img_w, img_h = img_info['width'], img_info['height']

        img_path = os.path.join(self.images_path, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        image = image.resize((self.image_size, self.image_size))
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        labels = []
        for label in self.labels.get(img_info['id'], []):
            if label['iscrowd']:
                continue
            x, y, w, h = label['bbox']
            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            class_idx = self.cat_to_idx[label['category_id']]
            labels.append([class_idx, cx, cy, w / img_w, h / img_h])

        labels = torch.tensor(labels, dtype=torch.float32) if labels else torch.zeros((0, 5))

        return image, labels

    def __getitem__(self, index):
        # Skip past corrupt/unreadable images rather than crashing the whole worker.
        for _ in range(len(self.images)):
            try:
                return self._load(index)
            except (OSError, ValueError) as e:
                print(f"[coco_dataset] skipping idx {index} ({self.images[index].get('file_name')}): {e}")
                index = (index + 1) % len(self.images)
        raise RuntimeError("No loadable images in dataset.")


def yolo_collate(batch):
    # images stack into [B, 3, 416, 416]; labels stay as a list since N varies per image
    images = torch.stack([b[0] for b in batch], dim=0)
    labels = [b[1] for b in batch]
    return images, labels
