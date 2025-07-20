# Advanced-Object-Counting-and-Classification-with-YOLOv8-and-ESRGAN-NHA
Smart traffic monitoring on highways  Entry/exit analytics for toll booths  Infrastructure planning and traffic policy design  Surveillance enhancement with high-resolution classification.
# 🚧 Smart Traffic Analytics System for NHA (National Highway Authority)

This is a project I worked on for **National Highway Authority (NHA)** to monitor vehicles on highways using deep learning and computer vision.

It includes vehicle detection, IN/OUT counting, super-resolution to enhance object crops, and classification to get detailed stats like what kind of vehicles are passing through (car, bus, truck, etc.).

---

## 🔧 What it does:

- Detects vehicles in a video using **YOLOv8**
- Tracks them across frames
- Counts how many went **IN** or **OUT** of a defined region
- Applies **ESRGAN super-resolution** to enhance cropped vehicle images
- Classifies them using a trained YOLOv8 classifier model (`best.pt`)
- Shows analytics per class (e.g., how many trucks went IN, how many cars went OUT)
- Outputs a final video with all the visual overlays

---

## 🛠 Tech Used:

- **Python**
- **YOLOv8 (Ultralytics)** for detection & tracking
- **PyTorch** for ESRGAN super-resolution
- **OpenCV** for video reading/writing
- **PIL, NumPy** for image handling
- **CUDA GPU** for faster processing

---

## 🗂 Folder Structure:
