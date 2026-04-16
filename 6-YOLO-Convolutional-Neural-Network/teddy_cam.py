from ultralytics import YOLO

model = YOLO("runs/detect/train5/weights/best.pt")

model.track(source=0, show=True) 