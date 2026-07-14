"""Bonus module: cartoonify an image using edge + bilateral filtering."""
import cv2
import argparse


def cartoonify(image_path: str, out_path: str = "outputs/cartoon.jpg"):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img, 9, 250, 250)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    cv2.imwrite(out_path, cartoon)
    print(f"Saved cartoon image to {out_path}")
    return cartoon


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", "-i", required=True)
    args = parser.parse_args()
    cartoonify(args.image)
