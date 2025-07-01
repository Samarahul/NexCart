from ultralytics import YOLO

# Load the model
model = YOLO('yolov8n.pt')  # Choose the model based on your needs

# Start training
model.train(
    data='data.yaml',    # Path to your data.yaml file
    epochs=4,             # Number of epochs
    imgsz=640               # Image size (you can adjust this)
)
