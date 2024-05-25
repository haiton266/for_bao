import cv2
from ultralytics import YOLO
import time
# Load the YOLOv8 model
model = YOLO("model/yolov8n.pt")

# Open the video file
video_path = "video/hota10.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    start = time.time()
    # Read a frame from the video
    success, frame = cap.read()

    if success:

        results = model.predict(frame)
        end = time.time()
        print(f"Current FPS: {1/(end - start)}")
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
