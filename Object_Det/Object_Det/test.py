import cv2
import logging
from ultralytics import YOLO

# Suppress YOLO logging by setting the logging level to ERROR
logging.getLogger().setLevel(logging.ERROR)

# Load the trained YOLO model (replace with the path to your trained model)
model = YOLO('runs/detect/train3/weights/best.pt')  # Use your trained model

# Initialize the webcam or any video capture (0 means default camera)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open video stream")
    exit()

# Loop to continuously get frames from the camera and detect objects
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image")
        break

    # Run YOLOv8 inference on the frame
    results = model(frame, stream=True)  # Run detection on the frame

    # Loop through the results and draw boxes and labels
    for result in results:
        for box in result.boxes:  # Each result.boxes contains detected boxes
            # Get box coordinates (x1, y1, x2, y2)
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to int for OpenCV
            
            # Get confidence score and class ID
            conf = box.conf[0]  # Confidence score
            cls = int(box.cls[0])  # Class ID (integer)

            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Get the label (class name) based on class ID
            label = model.names[cls]  # Get class name from the model

            # Display the label and confidence on the frame
            label_text = f"{label} ({conf:.2f})"
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the resulting frame with detections
    cv2.imshow('YOLO Object Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
