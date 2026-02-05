import os

class Config:
    FORCED_CAMERA_INDEX = 0 
    YOLO_MODEL_PATH = "models/yolov8n.pt"
    VIOLENCE_MODEL_PATH = "models/violencemagicvit.h5"
    YOLO_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    FPS = 15
    CONFIDENCE_THRESHOLD = 0.5
    VIOLENCE_THRESHOLD = 0.7
    
    # EXACTLY MATCHING YOUR MODEL:
    SEQUENCE_LENGTH = 20      # Must be 20
    FRAME_SIZE = (112, 112)   # Must be (112, 112)
    
    SAVE_VIDEO = True
    OUTPUT_DIR = "output"
    
    def create_directories(self):
        os.makedirs("models", exist_ok=True)
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
    
    def download_yolo_model(self):
        if not os.path.exists(self.YOLO_MODEL_PATH):
            print("Downloading YOLOv8 model...")
            import urllib.request
            urllib.request.urlretrieve(self.YOLO_MODEL_URL, self.YOLO_MODEL_PATH)
            print("YOLOv8 model downloaded successfully!")

config = Config()
config.create_directories()
