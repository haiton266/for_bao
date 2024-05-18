from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('yolov8n.pt')

# Run inference on 'bus.jpg' with arguments
model.predict('https://ultralytics.com/images/bus.jpg',
              save=True, imgsz=640, conf=0.5)
