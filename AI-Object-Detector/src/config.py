"""Central configuration for AI Object Detector."""
import os
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "yolov8n.pt")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
IMAGES_DIR = os.path.join(BASE_DIR, "images")
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")

CONFIDENCE_THRESHOLD = 0.45
IOU_THRESHOLD = 0.45
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (255, 255, 255)
FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2

WEBCAM_INDEX = 0
EXIT_KEY = "q"

os.makedirs(OUTPUT_DIR, exist_ok=True)
