"""Run YOLOv8 object detection on a single image."""
import argparse
import os
import sys
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config
from src.utils import load_model, draw_detections, safe_output_path, logger


def detect_image(image_path: str, save: bool = True, show: bool = True):
    if not os.path.exists(image_path):
        logger.error(f"Image not found: {image_path}")
        raise FileNotFoundError(f"Image not found: {image_path}")

    frame = cv2.imread(image_path)
    if frame is None:
        logger.error(f"OpenCV could not read image: {image_path}")
        raise ValueError(f"Invalid or corrupted image: {image_path}")

    model = load_model()
    results = model.predict(frame, conf=config.CONFIDENCE_THRESHOLD,
                             iou=config.IOU_THRESHOLD, device=config.DEVICE, verbose=False)
    frame, count = draw_detections(frame, results)
    logger.info(f"Detected {count} object(s) in {os.path.basename(image_path)}")

    if save:
        out_name = f"detected_{os.path.basename(image_path)}"
        out_path = safe_output_path(out_name)
        cv2.imwrite(out_path, frame)
        logger.info(f"Saved result to {out_path}")

    if show:
        cv2.imshow("AI Object Detector - Image", frame)
        logger.info("Press any key to close the window.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return frame, count


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Image Object Detection")
    parser.add_argument("--image", "-i", required=True, help="Path to input image")
    parser.add_argument("--no-show", action="store_true", help="Do not open a display window")
    parser.add_argument("--no-save", action="store_true", help="Do not save output image")
    args = parser.parse_args()

    try:
        detect_image(args.image, save=not args.no_save, show=not args.no_show)
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
