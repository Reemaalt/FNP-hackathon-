########## main ##########
"""
This file contain the main execution cycle of the project 
"""

# Import needed libraries
from utils import load_model, detect_in_image, detect_in_video, detect_and_report
from ultralytics import YOLO

# Main
if __name__ == "__main__":
    
    # Load the model
    model_path = "Models/yolov8m.pt"
    yolo_model = load_model(model_path)
    
    # Example usage for detect in an image
    #detect_in_image("Images/test_image_1.jpg", yolo_model)
    
    # Example usage for detect in a video
    #detect_in_video("Videos/test_video_1.mp4", yolo_model)

    # Example usage for detect and report
    detect_and_report("Videos/test_video_1.mp4", yolo_model)
    