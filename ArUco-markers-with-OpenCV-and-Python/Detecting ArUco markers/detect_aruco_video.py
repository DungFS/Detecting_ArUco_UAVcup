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
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Screen ArUco scanner for UAVcup – capture right half of screen,
# auto-detect multiple dictionaries, lock & scan N markers with pose.

import time
import sys
from collections import deque
import numpy as np
import cv2
import pyautogui
import argparse

# --------------- CLI ---------------
ap = argparse.ArgumentParser(
    description="ArUco scanner from SCREEN (right half) with auto-dict + pose"
)
ap.add_argument("--dict", default="",
                help="Force ArUco dictionary (e.g. DICT_4X4_50). "
                     "Leave empty to AUTO-detect & lock.")
ap.add_argument("--width", type=int, default=800,
                help="Resize width for processing (default: 800)")
ap.add_argument("--target", type=int, default=20,
                help="Number of unique markers to collect (default: 20)")
ap.add_argument("--stable-frames", type=int, default=3,
                help="Frames required to confirm 1 marker (default: 3)")
ap.add_argument("--calib", default="",
                help="camera_params.npz (cameraMatrix, distCoeffs) for pose")
ap.add_argument("--marker", type=float, default=0.0,
                help="Marker side length in meters (pose). 0 = disable pose")
ap.add_argument("--draw-rejected", action="store_true",
                help="Draw rejected candidates (debug)")
args = ap.parse_args()

# --------------- ArUco dicts ---------------
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

def make_fast_params():
    p = cv2.aruco.DetectorParameters()
    p.adaptiveThreshWinSizeMin = 3
    p.adaptiveThreshWinSizeMax = 23
    p.adaptiveThreshWinSizeStep = 10
    p.adaptiveThreshConstant = 7
    p.minMarkerPerimeterRate = 0.02
    p.maxMarkerPerimeterRate = 4.0
    p.minCornerDistanceRate = 0.05
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return p

def load_calib(path):
    if not path:
        return None, None
    try:
        d = np.load(path)
        return d["cameraMatrix"], d["distCoeffs"]
    except Exception as e:
        print("[WARN] Cannot load calib:", e)
        return None, None

def draw_fps(img, fps, dict_name, frame_markers, collected, target, locked, auto_mode):
    if auto_mode and not locked:
        dtxt = "AUTO"
    else:
        dtxt = dict_name + (" (locked)" if locked else "")
    txt = f"FPS:{fps:.1f} dict:{dtxt} frame:{frame_markers} collected:{collected}/{target}"
    cv2.putText(img, txt, (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

def main():
    # Dict mode
    forced_dict_name = args.dict.strip().upper()
    auto_mode = (forced_dict_name == "")

    if not auto_mode and forced_dict_name not in ARUCO_DICTS:
        print("[ERROR] Unknown dict:", forced_dict_name)
        print("Valid:", ", ".join(ARUCO_DICTS.keys()))
        sys.exit(1)

    # OpenCV optimize
    cv2.setUseOptimized(True)
    try:
        cv2.setNumThreads(4)
    except Exception:
        pass

    params = make_fast_params()

    # Prepare detectors
    detectors = {}
    if auto_mode:
        for name, dval in ARUCO_DICTS.items():
            dd = cv2.aruco.getPredefinedDictionary(dval)
            detectors[name] = cv2.aruco.ArucoDetector(dd, params)
        active_dict_name = "AUTO"
        locked = False
    else:
        dd = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[forced_dict_name])
        detectors[forced_dict_name] = cv2.aruco.ArucoDetector(dd, params)
        active_dict_name = forced_dict_name
        locked = True

    # Pose
    K, D = load_calib(args.calib)
    use_pose = (K is not None and D is not None and args.marker > 0.0)

    # Screen + ROI: right half (AirServer)
    screen_w, screen_h = pyautogui.size()
    roi_left = screen_w // 2
    roi_top = 0
    roi_width = screen_w - roi_left
    roi_height = screen_h

    print(f"[INFO] Screen: {screen_w}x{screen_h}")
    print(f"[INFO] Capturing RIGHT half: left={roi_left}, top={roi_top}, w={roi_width}, h={roi_height}")
    print("[INFO] Put AirServer window in the right half of the screen.")
    print(f"[INFO] Mode: {'AUTO dict' if auto_mode else 'FORCED ' + forced_dict_name}")
    print(f"[INFO] Pose: {'ON' if use_pose else 'OFF'}")

    # State
    target_count = args.target
    stable_required = args.stable_frames
    id_stability = {}
    final_ids = []
    recent_ids = deque(maxlen=30)

    frames = 0
    t0 = time.time()
    fps = 0.0

    pyautogui.PAUSE = 0
    pyautogui.FAILSAFE = False

    print("[INFO] Screen ArUco scanner running… Press 'q' to quit.")

    while True:
        # Screenshot ROI (right half)
        img = pyautogui.screenshot(region=(roi_left, roi_top, roi_width, roi_height))
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Resize for processing
        if args.width > 0:
            h, w = frame.shape[:2]
            if w != args.width:
                new_h = int(h * (args.width / float(w)))
                frame = cv2.resize(frame, (args.width, new_h),
                                   interpolation=cv2.INTER_LINEAR)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect markers (auto-dict or forced-dict)
        corners = ids = rejected = None

        if auto_mode and not locked:
            best_name = None
            best_count = 0
            best_c = best_i = best_r = None
            for name, det in detectors.items():
                c, i, r = det.detectMarkers(gray)
                n = 0 if i is None else len(i)
                if n > best_count:
                    best_count = n
                    best_name = name
                    best_c, best_i, best_r = c, i, r
            if best_count > 0:
                active_dict_name = best_name
                corners, ids, rejected = best_c, best_i, best_r
                locked = True
                print(f"[INFO] Auto-detected dict: {active_dict_name} (locked)")
            # nếu chưa thấy marker nào, corners/ids/rejected vẫn None
        else:
            det = detectors[active_dict_name]
            corners, ids, rejected = det.detectMarkers(gray)

        out = frame.copy()
        frame_markers = 0
        current_ids = set()

        if ids is not None and len(ids) > 0:
            frame_markers = len(ids)
            cv2.aruco.drawDetectedMarkers(out, corners, ids)
            current_ids = set(int(m) for m in ids.flatten())

            # Stability logic
            for mid in current_ids:
                id_stability[mid] = id_stability.get(mid, 0) + 1
                if id_stability[mid] == stable_required and mid not in final_ids:
                    final_ids.append(mid)
                    recent_ids.append(mid)
                    print(f"[INFO] Confirmed marker {mid}. Progress: {len(final_ids)}/{target_count}")

            for mid in list(id_stability.keys()):
                if mid not in current_ids:
                    id_stability[mid] = 0

            # Pose or only IDs
            if use_pose:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, args.marker, K, D
                )
                for i, mid in enumerate(ids.flatten()):
                    rvec, tvec = rvecs[i][0], tvecs[i][0]
                    cv2.drawFrameAxes(out, K, D, rvec, tvec, args.marker / 2.0)
                    R, _ = cv2.Rodrigues(rvec)
                    yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
                    c = corners[i][0].mean(axis=0).astype(int)
                    txt = f"ID:{int(mid)} x:{tvec[0]:.2f} y:{tvec[1]:.2f} z:{tvec[2]:.2f}m yaw:{yaw:.1f}"
                    cv2.putText(out, txt, (c[0] - 140, c[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            else:
                for (markerCorner, markerID) in zip(corners, ids.flatten()):
                    pts = markerCorner.reshape((4, 2)).astype(int)
                    cX = int((pts[0,0] + pts[2,0]) / 2.0)
                    cY = int((pts[0,1] + pts[2,1]) / 2.0)
                    cv2.circle(out, (cX, cY), 4, (0,0,255), -1)
                    cv2.putText(out, f"{int(markerID)}",
                                (pts[0,0], pts[0,1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            # Stop when collected enough markers
            if len(final_ids) >= target_count:
                print("[INFO] Collected all markers.")
                print("RESULT IDs:", final_ids)
                cv2.putText(out, "DONE: all markers collected", (16, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.imshow("ArUco Screen Scanner", out)
                cv2.waitKey(2000)
                cv2.destroyAllWindows()
                print("[INFO] Final IDs:", final_ids)
                return

        # Draw rejected (optional)
        if args.draw_rejected and rejected is not None:
            for rc in rejected:
                pts = rc.reshape(-1,2).astype(int)
                for i in range(4):
                    cv2.line(out,
                             tuple(pts[i]),
                             tuple(pts[(i+1)%4]),
                             (0,0,255), 1)

        # FPS
        frames = getattr(main, "_frames", 0) + 1
        main._frames = frames
        if frames % 10 == 0:
            now = time.time()
            fps = 10.0 / (now - getattr(main, "_t0", time.time()))
            main._t0 = now
        else:
            fps = getattr(main, "_fps", 0.0)
        main._fps = fps

        draw_fps(out, fps, active_dict_name, frame_markers,
                 len(final_ids), target_count, locked, auto_mode)

        cv2.putText(out, f"SEEN: {final_ids}", (16, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)

        cv2.imshow("ArUco Screen Scanner (Right Half)", out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print("[INFO] Final IDs:", final_ids)

if __name__ == "__main__":
    if not hasattr(cv2, "aruco"):
        print("[ERROR] OpenCV build missing 'aruco'. Install: pip install opencv-contrib-python")
        sys.exit(1)
    main()
