from ultralytics import YOLO

# Load the trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Run prediction
results = model("road.jpg")

# Display the result
results[0].show()