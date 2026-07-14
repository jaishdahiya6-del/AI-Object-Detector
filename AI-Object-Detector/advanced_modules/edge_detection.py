"""Bonus module: Canny edge detection on an image."""
import cv2
import argparse


def detect_edges(image_path: str, out_path: str = "outputs/edges.jpg"):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    cv2.imwrite(out_path, edges)
    print(f"Saved edge map to {out_path}")
    return edges


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", "-i", required=True)
    args = parser.parse_args()
    detect_edges(args.image)
