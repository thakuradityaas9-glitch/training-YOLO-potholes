from ultralytics import YOLO

# Load YOLO11 nano pretrained model
model = YOLO("yolo11n.pt")

# Train on our road-damage dataset
model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device=0
)