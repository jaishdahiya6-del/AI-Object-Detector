# 🎯 AI Object Detector

**Real-time object detection using YOLOv8 + OpenCV — webcam, images, and video, with an optional GUI and a browser-deployable Streamlit app.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

> Add a real banner image here later: `docs/banner.png`

---

## ✨ Features

- Detect multiple objects with class label + confidence score
- Bounding boxes drawn live on frame
- Real-time FPS counter
- Works on **webcam**, **uploaded images**, and **video files**
- Saves annotated output images/videos to `outputs/`
- Keyboard exit (`q`) and snapshot key (`s`) in live modes
- Auto GPU (CUDA) detection, falls back to CPU
- Optional Tkinter desktop GUI
- Optional Streamlit web app for a browser-based live demo
- Bonus OpenCV modules: face detection, edge detection, cartoon/sketch filters, motion detection

---

## 📁 Folder Structure

```
AI-Object-Detector/
│
├── models/               # yolov8n.pt (auto-downloaded on first run)
├── images/                # put test images here
├── videos/                 # put test videos here
├── outputs/                # detection results are saved here
├── src/
│   ├── detect_image.py
│   ├── detect_video.py
│   ├── detect_webcam.py
│   ├── utils.py
│   └── config.py
├── advanced_modules/        # bonus OpenCV scripts
├── notebooks/
├── main.py                  # unified CLI entry point
├── gui.py                   # Tkinter desktop GUI
├── app.py                   # Streamlit web app (for live deployment)
├── requirements.txt
├── setup.py
├── LICENSE
├── .gitignore
└── README.md
```

---

## 🔧 Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/AI-Object-Detector.git
cd AI-Object-Detector
```

### 2. Create a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

The first time you run detection, `yolov8n.pt` weights are downloaded
automatically into `models/` — no manual step needed.

---

## ▶️ Usage

**Image detection**
```bash
python main.py --mode image --source images/sample.jpg
```

**Video detection**
```bash
python main.py --mode video --source videos/sample.mp4
```

**Webcam (real-time)**
```bash
python main.py --mode webcam
```
Press `q` to quit, `s` to save a snapshot.

**Desktop GUI**
```bash
python main.py --mode gui
```

**Web app (Streamlit)**
```bash
streamlit run app.py
```

You can also call each script directly, e.g. `python src/detect_image.py -i images/sample.jpg`.

---

## 🌍 Deploying It Live (via GitHub)

The webcam mode needs a local camera and OpenCV windows, so it can't run on a
normal web server. For a **live, shareable demo**, deploy `app.py`
(image/video upload in the browser) using Streamlit Community Cloud — free,
and connects straight to your GitHub repo.

### Step 1 — Push the project to GitHub
```bash
git init
git add .
git commit -m "Initial Commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/AI-Object-Detector.git
git push -u origin main
```
- `git init` — start a local repository
- `git add .` — stage every file
- `git commit` — save a snapshot with a message
- `git branch -M main` — rename default branch to `main`
- `git remote add origin ...` — link to your GitHub repo (create it empty on
  github.com first, no README/license, to avoid merge conflicts)
- `git push -u origin main` — upload everything

### Step 2 — Deploy on Streamlit Community Cloud
1. Go to https://share.streamlit.io and sign in with GitHub
2. Click **New app** → select `AI-Object-Detector` → branch `main` → main file `app.py`
3. Click **Deploy** — Streamlit installs `requirements.txt` and hosts it at
   a public URL like `https://your-app.streamlit.app`
4. Share that link — anyone can upload an image/video and see live detections
   in their browser, no install needed

### Alternative hosts
- **Hugging Face Spaces** (supports Streamlit/Gradio, free GPU tier available)
- **Render / Railway** for a Dockerized Flask/Streamlit deployment
- GitHub Pages **cannot** run Python — it's static hosting only, so it isn't
  an option for this project directly.

---

## ✅ Testing Checklist

- [ ] Webcam opens and shows live detections
- [ ] Image detection draws correct boxes + labels
- [ ] Video detection processes all frames without crashing
- [ ] FPS counter updates and looks reasonable
- [ ] Output image/video saved correctly in `outputs/`
- [ ] Confidence scores display next to labels
- [ ] Multiple object classes detected correctly
- [ ] `q` exits webcam/video windows cleanly
- [ ] GPU used automatically when CUDA available, CPU otherwise
- [ ] Errors (missing file, bad camera) show clear messages, not crashes

---

## 🚀 Future Improvements

- Custom-trained YOLOv8 model for domain-specific objects
- Multi-camera support
- REST API (FastAPI) for remote inference
- Docker image for one-command deployment
- Object tracking (ByteTrack/DeepSORT) across video frames

---

## 🤝 Contributing

Pull requests are welcome. For major changes, open an issue first to discuss
what you'd like to change.

## 📜 License

MIT — see [LICENSE](LICENSE).

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
- [Streamlit](https://streamlit.io/)

## 🏷️ Suggested GitHub Topics

`python` `opencv` `yolov8` `object-detection` `computer-vision` `deep-learning`
`streamlit` `real-time-detection` `pytorch` `ai`
