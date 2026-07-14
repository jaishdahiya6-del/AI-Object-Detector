"""Shared utility helpers: logging, FPS counter, drawing, model loader."""
import logging
import time
import os
import cv2
from ultralytics import YOLO

from src import config


def setup_logger(name: str = "AIObjectDetector") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


logger = setup_logger()


class FPSCounter:
    """Simple rolling FPS counter."""

    def __init__(self):
        self._prev_time = time.time()
        self.fps = 0.0

    def update(self) -> float:
        now = time.time()
        delta = now - self._prev_time
        self._prev_time = now
        self.fps = 1.0 / delta if delta > 0 else 0.0
        return self.fps


def load_model(model_path: str = config.MODEL_PATH) -> YOLO:
    """Load a YOLOv8 model, auto-downloading weights if missing."""
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        if not os.path.exists(model_path):
            logger.info("Model weights not found locally. Downloading yolov8n.pt ...")
            model = YOLO("yolov8n.pt")
            model.save(model_path)
        else:
            model = YOLO(model_path)
        model.to(config.DEVICE)
        logger.info(f"Model loaded successfully on device: {config.DEVICE.upper()}")
        return model
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        raise


def draw_detections(frame, results):
    """Draw bounding boxes + labels on a frame. Returns (frame, detection_count)."""
    count = 0
    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = f"{r.names[cls_id]} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), config.BOX_COLOR, 2)
            (tw, th), _ = cv2.getTextSize(label, config.FONT, config.FONT_SCALE, config.FONT_THICKNESS)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw, y1), config.BOX_COLOR, -1)
            cv2.putText(frame, label, (x1, y1 - 5), config.FONT, config.FONT_SCALE,
                        config.TEXT_COLOR, config.FONT_THICKNESS)
            count += 1
    return frame, count


def draw_fps(frame, fps: float):
    cv2.putText(frame, f"FPS: {fps:.1f}", (15, 30), config.FONT, 0.8, (0, 255, 255), 2)
    return frame


def safe_output_path(filename: str) -> str:
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    return os.path.join(config.OUTPUT_DIR, filename)
