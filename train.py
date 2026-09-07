from ultralytics import YOLO

model = YOLO("yolo11s.pt")

model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=-1,
    device=0,
    patience=20,
    workers=8,
    project="runs/model_a",
    name="yolo11s_pavement_damage"
)