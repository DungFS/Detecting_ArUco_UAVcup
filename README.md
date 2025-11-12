# 🚀 Detecting ArUco Markers for UAVcup 2025

This project is part of **UAVcup 2025**, providing a real-time vision system for UAVs to detect and identify **ArUco markers** from video streams (DJI camera, USB, or UDP).  
It is designed for waypoint automation and visual navigation tasks using Python + OpenCV.

---

## ✈️ Features
- Real-time ArUco marker detection (OpenCV 4.8+)  
- Works with webcam / RTSP / RTMP / UDP (DJI feed)  
- Auto dictionary detection and pose estimation (x, y, z, yaw)  
- Adjustable parameters for low-light or grayscale prints  
- Ready for PX4 / MAVSDK integration for autonomous flight

---

## ⚙️ Quick Start
```bash
pip install -r requirements.txt
python aruco_scanner.py --src 0
