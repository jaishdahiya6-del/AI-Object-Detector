# Advanced / Bonus Modules

Standalone OpenCV extras you can run independently of the main YOLO pipeline:

| Module | Run |
|---|---|
| Face Detection | `python advanced_modules/face_detection.py` |
| Edge Detection | `python advanced_modules/edge_detection.py -i images/sample.jpg` |
| Cartoon Filter | `python advanced_modules/cartoon_filter.py -i images/sample.jpg` |
| Sketch Filter | `python advanced_modules/sketch_filter.py -i images/sample.jpg` |
| Motion Detection | `python advanced_modules/motion_detection.py` |

More modules (QR/Barcode scanner, pose detection, hand tracking, vehicle/people
counter, license-plate detection, color detection, background removal) follow
the same pattern — each is a small, self-contained script built on OpenCV or
Ultralytics. Ask and they can be added the same way.
