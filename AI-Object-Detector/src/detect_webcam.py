"""Run YOLOv8 real-time object detection on a webcam feed."""
import argparse
import sys
import os
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config
from src.utils import load_model, draw_detections, draw_fps, FPSCounter, safe_output_path, logger


def detect_webcam(camera_index: int = config.WEBCAM_INDEX, save_snapshots: bool = False):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error(f"Could not access webcam at index {camera_index}. "
                      f"Check camera permissions or try a different index.")
        raise RuntimeError("Webcam not accessible")

    model = load_model()
    fps_counter = FPSCounter()
    logger.info(f"Webcam started. Press '{config.EXIT_KEY.upper()}' to quit, 's' to save a snapshot.")

    snap_id = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to grab frame from webcam. Retrying...")
                continue

            results = model.predict(frame, conf=config.CONFIDENCE_THRESHOLD,
                                     iou=config.IOU_THRESHOLD, device=config.DEVICE, verbose=False)
            frame, count = draw_detections(frame, results)
            fps = fps_counter.update()
            frame = draw_fps(frame, fps)
            cv2.putText(frame, f"Objects: {count}", (15, 60), config.FONT, 0.8, (255, 200, 0), 2)

            cv2.imshow("AI Object Detector - Webcam", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(config.EXIT_KEY):
                logger.info("Exit key pressed. Closing webcam.")
                break
            elif key == ord("s"):
                snap_id += 1
                path = safe_output_path(f"webcam_snapshot_{snap_id}.jpg")
                cv2.imwrite(path, frame)
                logger.info(f"Snapshot saved: {path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Webcam Object Detection")
    parser.add_argument("--camera", "-c", type=int, default=config.WEBCAM_INDEX, help="Webcam index")
    args = parser.parse_args()

    try:
        detect_webcam(args.camera)
    except Exception as e:
        logger.error(f"Webcam detection failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
