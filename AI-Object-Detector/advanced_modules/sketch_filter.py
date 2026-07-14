"""Bonus module: pencil-sketch effect for an image."""
import cv2
import argparse


def sketchify(image_path: str, out_path: str = "outputs/sketch.jpg"):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted = 255 - gray
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blurred, scale=256)
    cv2.imwrite(out_path, sketch)
    print(f"Saved sketch image to {out_path}")
    return sketch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", "-i", required=True)
    args = parser.parse_args()
    sketchify(args.image)
