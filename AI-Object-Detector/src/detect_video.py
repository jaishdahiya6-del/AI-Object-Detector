"""Run YOLOv8 object detection on a video file, with output saving."""
import argparse
import os
import sys
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config
from src.utils import load_model, draw_detections, draw_fps, FPSCounter, safe_output_path, logger


def detect_video(video_path: str, save: bool = True, show: bool = True):
    if not os.path.exists(video_path):
        logger.error(f"Video not found: {video_path}")
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"OpenCV could not open video: {video_path}")
        raise ValueError(f"Invalid or corrupted video: {video_path}")

    model = load_model()
    fps_counter = FPSCounter()

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    writer = None
    if save:
        out_name = f"detected_{os.path.basename(video_path)}"
        out_path = safe_output_path(out_name)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, src_fps, (width, height))
        logger.info(f"Saving annotated video to {out_path}")

    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(frame, conf=config.CONFIDENCE_THRESHOLD,
                                     iou=config.IOU_THRESHOLD, device=config.DEVICE, verbose=False)
            frame, count = draw_detections(frame, results)
            fps = fps_counter.update()
            frame = draw_fps(frame, fps)
            frame_count += 1

            if writer is not None:
                writer.write(frame)

            if show:
                cv2.imshow("AI Object Detector - Video", frame)
                if cv2.waitKey(1) & 0xFF == ord(config.EXIT_KEY):
                    logger.info("Exit key pressed. Stopping video detection.")
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    logger.info(f"Processed {frame_count} frames.")


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Video Object Detection")
    parser.add_argument("--video", "-v", required=True, help="Path to input video file")
    parser.add_argument("--no-show", action="store_true", help="Do not open a display window")
    parser.add_argument("--no-save", action="store_true", help="Do not save output video")
    args = parser.parse_args()

    try:
        detect_video(args.video, save=not args.no_save, show=not args.no_show)
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
