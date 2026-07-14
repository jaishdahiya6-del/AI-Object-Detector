"""Tkinter GUI for AI Object Detector.

Buttons: Open Image, Open Video, Start Webcam, Save Result, Exit.
"""
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

from src.utils import logger

_last_result_path = {"path": None}


def launch_gui():
    root = tk.Tk()
    root.title("AI Object Detector")
    root.geometry("420x320")
    root.resizable(False, False)

    tk.Label(root, text="AI Object Detection System", font=("Segoe UI", 16, "bold")).pack(pady=18)
    status_var = tk.StringVar(value="Ready.")

    def set_status(msg):
        status_var.set(msg)
        root.update_idletasks()

    def open_image():
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return
        set_status(f"Detecting objects in {os.path.basename(path)} ...")

        def run():
            try:
                from src.detect_image import detect_image
                _, count = detect_image(path, save=True, show=True)
                _last_result_path["path"] = path
                set_status(f"Done. {count} object(s) detected. Saved to outputs/.")
            except Exception as e:
                logger.error(f"GUI image detection failed: {e}")
                messagebox.showerror("Error", str(e))
                set_status("Error during image detection.")

        threading.Thread(target=run, daemon=True).start()

    def open_video():
        path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
        if not path:
            return
        set_status(f"Processing video {os.path.basename(path)} ... (press Q in window to stop)")

        def run():
            try:
                from src.detect_video import detect_video
                detect_video(path, save=True, show=True)
                _last_result_path["path"] = path
                set_status("Video processed. Saved to outputs/.")
            except Exception as e:
                logger.error(f"GUI video detection failed: {e}")
                messagebox.showerror("Error", str(e))
                set_status("Error during video detection.")

        threading.Thread(target=run, daemon=True).start()

    def start_webcam():
        set_status("Starting webcam ... (press Q in window to stop)")

        def run():
            try:
                from src.detect_webcam import detect_webcam
                detect_webcam()
                set_status("Webcam session ended.")
            except Exception as e:
                logger.error(f"GUI webcam detection failed: {e}")
                messagebox.showerror("Error", str(e))
                set_status("Error starting webcam.")

        threading.Thread(target=run, daemon=True).start()

    def save_result():
        from src import config
        messagebox.showinfo("Save Result", f"All results are auto-saved to:\n{config.OUTPUT_DIR}")

    def exit_app():
        root.destroy()

    btn_style = {"width": 26, "height": 2, "font": ("Segoe UI", 10)}
    tk.Button(root, text="Open Image", command=open_image, **btn_style).pack(pady=4)
    tk.Button(root, text="Open Video", command=open_video, **btn_style).pack(pady=4)
    tk.Button(root, text="Start Webcam", command=start_webcam, **btn_style).pack(pady=4)
    tk.Button(root, text="Save Result", command=save_result, **btn_style).pack(pady=4)
    tk.Button(root, text="Exit", command=exit_app, **btn_style).pack(pady=4)

    tk.Label(root, textvariable=status_var, fg="gray20").pack(pady=10)

    root.mainloop()


if __name__ == "__main__":
    launch_gui()
