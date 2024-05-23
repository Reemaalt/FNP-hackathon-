########## utils ##########
"""
This file contain all functions which will be used during the project
"""

# Import needed libraries
from ultralytics import YOLO
import numpy as np
import cv2

# Function to load YOLOv8 pretrained model
def load_model(model_path):
    # Load the YOLO model
    model = YOLO(model_path)
    return model

# Function to generate a unique color based on the class_id
def get_unique_color(class_id):
    np.random.seed(hash(class_id) % 255)
    color = np.random.randint(0, 255, size=(3,), dtype=np.uint8)
    return tuple(map(int, color))

# Function to detect objects in an image
def detect_in_image(image_path, model, show_confidence=False):
    # Load image
    image = cv2.imread(image_path)

    # Get results
    results = model.predict(image)
    result = results[0]

    # Loop through detected objects
    for box in result.boxes:
        class_id = result.names[box.cls[0].item()]
        cords = box.xyxy[0].tolist()
        cords = [round(x) for x in cords]
        conf = round(box.conf[0].item(), 2)

        # Get a unique color for the bounding box
        color = get_unique_color(class_id)

        # Draw a thin bounding box
        cv2.rectangle(image, (cords[0], cords[1]), (cords[2], cords[3]), color, 2)

        # Display class label and confidence
        if show_confidence:
            label = f"{class_id}: {conf}"
        else:
            label = f"{class_id}"
        cv2.putText(image, label, (cords[0], cords[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Display the image with bounding boxes
    cv2.imshow("Object Detection - Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to detect objects in a video
def detect_in_video(video_path, model, show_confidence=False):
    # Open video file
    cap = cv2.VideoCapture(video_path)

    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            break

        # Get results
        results = model.predict(frame)
        result = results[0]

        # Loop through detected objects
        for box in result.boxes:
            class_id = result.names[box.cls[0].item()]
            cords = box.xyxy[0].tolist()
            cords = [round(x) for x in cords]
            conf = round(box.conf[0].item(), 2)
            
            # Get a unique color for the bounding box
            color = get_unique_color(class_id)

            # Draw a thin bounding box
            cv2.rectangle(frame, (cords[0], cords[1]), (cords[2], cords[3]), color, 2)

            # Display class label and confidence
            if show_confidence:
                label = f"{class_id}: {conf}"
            else:
                label = f"{class_id}"
            cv2.putText(frame, label, (cords[0], cords[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Get a unique color for the bounding box
        color = get_unique_color(class_id)

        # Draw a thin bounding box
        cv2.rectangle(frame, (cords[0], cords[1]), (cords[2], cords[3]), color, 1)

        # Break the loop if 'q' or 'esc' is pressed
        cv2.imshow('Object Detection - Video', frame)
        key = cv2.waitKey(1)  
        if key == ord('q') or key == 27:  
            break

    # Release video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Function to detect objects in an image
def detect_and_report(video_path, model, show_table_only=True):
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    # Initialize tables count
    table_count = 0

    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            break

        # Get results
        result = model.predict(frame)[0]

        # Loop through detected objects
        for box in result.boxes:
            class_id = result.names[box.cls[0].item()]
            if(class_id != "dining table" and show_table_only==True):
                continue
            cords = box.xyxy[0].tolist()
            cords = [round(x) for x in cords]
            conf = round(box.conf[0].item(), 2)
            table_count += 1
            
            # Get a unique color for the bounding box
            color = get_unique_color(class_id)

            # Draw the bounding box
            cv2.rectangle(frame, (cords[0], cords[1]), (cords[2], cords[3]), color, 3)

            # Display class label and confidence
            label = f"{class_id}"
            if(class_id == "dining table"):
                label = f"Table {table_count}"
            cv2.putText(frame, label, (cords[0], cords[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
        # Reset table count
        table_count = 0

        # Break the loop if 'q' or 'esc' is pressed
        cv2.imshow('Object Detection - Video', frame)
        key = cv2.waitKey(1)  
        if key == ord('q') or key == 27:  
            break

    # Release video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()
