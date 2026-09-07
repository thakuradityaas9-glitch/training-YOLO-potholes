from ultralytics import YOLO
from PIL import Image

# Load our trained model
model = YOLO("runs/detect/train/weights/yolo11n.pt")

# Run prediction
results = model.predict(
    source="test.jpg",
    conf=0.25,
    save=True
)

# Display the first result
result_image = results[0].plot()

Image.fromarray(result_image).show()