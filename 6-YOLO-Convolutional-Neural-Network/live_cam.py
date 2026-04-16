import os
import shutil
import random
from ultralytics import YOLO


def split_dataset(base_dir, val_split=0.2, seed=42):
    images_dir = os.path.join(base_dir, "images")
    labels_dir = os.path.join(base_dir, "labels")

    for split in ["train", "val"]:
        os.makedirs(os.path.join(images_dir, split), exist_ok=True)
        os.makedirs(os.path.join(labels_dir, split), exist_ok=True)

    images = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
    random.seed(seed)
    random.shuffle(images)

    val_count = int(len(images) * val_split)
    splits = {"val": images[:val_count], "train": images[val_count:]}

    for split, files in splits.items():
        for fname in files:
            stem = os.path.splitext(fname)[0]

            shutil.move(os.path.join(images_dir, fname),
                        os.path.join(images_dir, split, fname))

            for lbl in os.listdir(labels_dir):
                if os.path.isfile(os.path.join(labels_dir, lbl)) and stem in lbl:
                    shutil.move(os.path.join(labels_dir, lbl),
                                os.path.join(labels_dir, split, lbl))
                    break

    print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}")
split_dataset("/Users/joehill/Developer/DeepLearning/5-YOLO-Convolutional-Neural-Network/project-1-at-2026-04-14-17-31-ada8e726")

model = YOLO("yolo26n.pt")

model.train(
    data="/Users/joehill/Developer/DeepLearning/5-YOLO-Convolutional-Neural-Network/teddy_dataset.yaml",
    epochs = 50,
    imgsz = 640,
    batch = 8,
    freeze = 10
)

model(source=0, show=True)  # source=0 = built-in camera, press Q to quit
