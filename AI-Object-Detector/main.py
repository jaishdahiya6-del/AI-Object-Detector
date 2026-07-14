"""AI Object Detector - unified command-line entry point.

Usage:
    python main.py --mode image --source images/sample.jpg
    python main.py --mode video --source videos/sample.mp4
    python main.py --mode webcam
    python main.py --mode gui
"""
import argparse
import sys

from src.utils import logger


def main():
    parser = argparse.ArgumentParser(description="AI Object Detection System (YOLOv8 + OpenCV)")
    parser.add_argument("--mode", "-m", choices=["image", "video", "webcam", "gui"],
                         required=True, help="Detection mode")
    parser.add_argument("--source", "-s", help="Path to image/video (required for image/video mode)")
    parser.add_argument("--no-show", action="store_true", help="Do not open a display window")
    parser.add_argument("--no-save", action="store_true", help="Do not save output")
    args = parser.parse_args()

    try:
        if args.mode == "image":
            if not args.source:
                parser.error("--source is required for image mode")
            from src.detect_image import detect_image
            detect_image(args.source, save=not args.no_save, show=not args.no_show)

        elif args.mode == "video":
            if not args.source:
                parser.error("--source is required for video mode")
            from src.detect_video import detect_video
            detect_video(args.source, save=not args.no_save, show=not args.no_show)

        elif args.mode == "webcam":
            from src.detect_webcam import detect_webcam
            detect_webcam()

        elif args.mode == "gui":
            from gui import launch_gui
            launch_gui()

    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
