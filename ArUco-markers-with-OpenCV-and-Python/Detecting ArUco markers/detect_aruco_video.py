#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ArUco scanner – OpenCV 4.8+ (no imutils), Python 3.10+

import argparse, time, sys
from collections import deque
import numpy as np
import cv2

# -------------------- CLI --------------------
ap = argparse.ArgumentParser(description="Real-time ArUco scanner (auto-dict, pose, FPS)")
ap.add_argument("--src", required=True, help="Video source: index (0/1/2) or URL (rtmp/rtsp/udp/file)")
ap.add_argument("--dict", default="", help="Force dictionary (e.g., DICT_4X4_50). Leave empty to auto-detect")
ap.add_argument("--calib", default="", help="Path to camera_params.npz (cameraMatrix, distCoeffs)")
ap.add_argument("--marker", type=float, default=0.0, help="Marker side length in meters (for pose). 0 = skip pose")
ap.add_argument("--draw-rejected", action="store_true", help="Draw rejected candidates (debug)")
ap.add_argument("--width", type=int, default=1000, help="Resize width for processing/display")
args = ap.parse_args()

# -------------------- DICTS --------------------
ARUCO_DICTS = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}

def make_params_relaxed():
    p = cv2.aruco.DetectorParameters()
    # Nới ngưỡng thích nghi để chịu ánh sáng xấu / giấy in xám
    p.adaptiveThreshWinSizeMin = 3
    p.adaptiveThreshWinSizeMax = 53
    p.adaptiveThreshWinSizeStep = 10
    p.adaptiveThreshConstant = 7
    # Nới tỉ lệ chu vi để bắt nhỏ/lớn hơn
    p.minMarkerPerimeterRate = 0.02
    p.maxMarkerPerimeterRate = 4.0
    # Góc/biên
    p.minCornerDistanceRate = 0.05
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return p

def open_source(src_str):
    # Cho phép truyền index (0/1/2) dạng string
    cap = None
    if src_str.isdigit():
        cap = cv2.VideoCapture(int(src_str))
    else:
        cap = cv2.VideoCapture(src_str)
    if not cap.isOpened():
        print("[ERROR] Cannot open source:", src_str)
        sys.exit(1)
    return cap

def load_calib(path):
    if not path:
        return None, None
    try:
        d = np.load(path)
        return d["cameraMatrix"], d["distCoeffs"]
    except Exception as e:
        print("[WARN] Cannot load calib:", e)
        return None, None

def draw_fps(img, fps, dict_name, count, locked):
    txt = f"FPS:{fps:.1f}  dict:{dict_name}{' (locked)' if locked else ''}  markers:{count}"
    cv2.putText(img, txt, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

def main():
    # Detector params
    base_params = make_params_relaxed()

    # Nếu user chỉ định dict, dùng luôn; nếu không sẽ auto-detect
    forced_dict_name = args.dict.strip().upper()
    dict_names_order = list(ARUCO_DICTS.keys())
    if forced_dict_name and forced_dict_name not in ARUCO_DICTS:
        print("[ERROR] Unknown dict:", forced_dict_name)
        print("Valid:", ", ".join(dict_names_order))
        sys.exit(1)

    # Chuẩn bị dict + detector
    def make_detector(dict_name):
        dd = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[dict_name])
        return dict_name, cv2.aruco.ArucoDetector(dd, base_params)

    if forced_dict_name:
        active_dict_name, detector = make_detector(forced_dict_name)
        locked = True
    else:
        # Auto mode: thử tất cả dicts mỗi frame cho đến khi bắt được ổn, rồi lock
        active_dict_name, detector = make_detector("DICT_4X4_50")  # khởi tạo tạm
        locked = False

    # Video
    cap = open_source(args.src)

    # Calib for pose
    K, D = load_calib(args.calib)
    use_pose = (K is not None and D is not None and args.marker > 0)

    # Thống kê
    seen_ids = []
    recent_ids = deque(maxlen=50)
    t0, frames = time.time(), 0
    fps = 0.0

    print("[INFO] Running… Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        # Resize theo width (giữ tỉ lệ)
        if args.width > 0:
            h, w = frame.shape[:2]
            if w != args.width:
                new_h = int(h * (args.width / float(w)))
                frame = cv2.resize(frame, (args.width, new_h), interpolation=cv2.INTER_LINEAR)

        # Dùng kênh xám + equalize để tăng tương phản
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # Auto-dict nếu chưa lock
        corners, ids, rejected = None, None, None
        if not locked and not forced_dict_name:
            best = ("", 0, None, None, None)  # (name, count, corners, ids, rejected)
            for name in dict_names_order:
                _, det = make_detector(name)
                c, i, r = det.detectMarkers(gray)
                n = 0 if i is None else len(i)
                if n > best[1]:
                    best = (name, n, c, i, r)
            active_dict_name = best[0] if best[1] > 0 else active_dict_name
            if best[1] > 0:
                # Nếu tìm thấy, lock vào dict tốt nhất sau khi thấy >= 5 frames đầu
                corners, ids, rejected = best[2], best[3], best[4]
                locked = True
                _, detector = make_detector(active_dict_name)
        else:
            corners, ids, rejected = detector.detectMarkers(gray)

        # Vẽ kết quả
        out = frame.copy()
        count = 0
        if ids is not None and len(ids) > 0:
            count = len(ids)
            cv2.aruco.drawDetectedMarkers(out, corners, ids)

            # Pose (nếu có calib + marker size)
            if use_pose:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, args.marker, K, D
                )
                for i, mid in enumerate(ids.flatten()):
                    rvec, tvec = rvecs[i][0], tvecs[i][0]
                    cv2.drawFrameAxes(out, K, D, rvec, tvec, args.marker/2)
                    # Yaw tham khảo
                    R, _ = cv2.Rodrigues(rvec)
                    yaw = np.degrees(np.arctan2(R[1,0], R[0,0]))
                    c = corners[i][0].mean(axis=0).astype(int)
                    txt = f"ID:{int(mid)} x:{tvec[0]:.2f} y:{tvec[1]:.2f} z:{tvec[2]:.2f}m yaw:{yaw:.1f}"
                    cv2.putText(out, txt, (c[0]-120, c[1]-20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            else:
                # Vẽ ID + tâm (nếu không dùng pose)
                for (markerCorner, markerID) in zip(corners, ids.flatten()):
                    pts = markerCorner.reshape((4, 2)).astype(int)
                    cX = int((pts[0,0] + pts[2,0]) / 2.0)
                    cY = int((pts[0,1] + pts[2,1]) / 2.0)
                    cv2.circle(out, (cX, cY), 4, (0,0,255), -1)
                    cv2.putText(out, f"{int(markerID)}", (pts[0,0], pts[0,1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            # Ghi log ID đã thấy (duy nhất, theo thứ tự lần đầu gặp)
            for mid in ids.flatten():
                recent_ids.append(int(mid))
                if int(mid) not in seen_ids:
                    seen_ids.append(int(mid))

        # Vẽ rejected (debug)
        if args.draw_rejected and rejected is not None:
            for rc in rejected:
                pts = rc.reshape(-1,2).astype(int)
                for i in range(4):
                    cv2.line(out, tuple(pts[i]), tuple(pts[(i+1)%4]), (0,0,255), 1)

        # FPS
        frames += 1
        if frames % 10 == 0:
            now = time.time()
            fps = 10.0 / (now - t0)
            t0 = now
        draw_fps(out, fps, active_dict_name, count, locked)

        # Hiển thị danh sách đã thấy
        cv2.putText(out, f"SEEN: {seen_ids}", (16, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

        cv2.imshow("ArUco Scanner", out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Kiểm tra module aruco
    if not hasattr(cv2, "aruco"):
        print("[ERROR] OpenCV build missing 'aruco'. Install: pip install opencv-contrib-python")
        sys.exit(1)
    main()
