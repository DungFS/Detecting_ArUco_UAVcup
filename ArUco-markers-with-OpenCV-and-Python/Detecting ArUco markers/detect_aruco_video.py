#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Screen ArUco scanner for UAVcup – capture right half of screen,
# auto-detect multiple dictionaries, lock & scan 16 markers, show map UI,
# and optionally record video of the scan.

import time
import sys
import os
from datetime import datetime
from collections import deque
import numpy as np
import cv2
import pyautogui
import argparse

# --------------- CLI ---------------
ap = argparse.ArgumentParser(
    description="ArUco scanner from SCREEN (right half) with auto-dict + pose + map UI + recording"
)
ap.add_argument("--dict", default="",
                help="Force ArUco dictionary (e.g. DICT_4X4_50). "
                     "Leave empty to AUTO-detect & lock.")
ap.add_argument("--width", type=int, default=800,
                help="Resize width for processing (default: 800)")
ap.add_argument("--target", type=int, default=16,
                help="Number of unique markers to collect (default: 16, max 16)")
ap.add_argument("--stable-frames", type=int, default=3,
                help="Frames required to confirm 1 marker (default: 3)")
ap.add_argument("--calib", default="",
                help="camera_params.npz (cameraMatrix, distCoeffs) for pose")
ap.add_argument("--marker", type=float, default=0.0,
                help="Marker side length in meters (pose). 0 = disable pose")
ap.add_argument("--draw-rejected", action="store_true",
                help="Draw rejected candidates (debug, not shown on map)")
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

MAX_SLOTS = 16  # đúng 16 marker

# --------------- Global cho UI + recording ---------------
BTN_X1 = BTN_Y1 = BTN_X2 = BTN_Y2 = 0  # nút Stop Rec

recording = True         # đang ghi video hay không
video_writer = None      # đối tượng VideoWriter
record_folder = None     # thư mục lưu cho lần chạy này
video_path = None        # đường dẫn file video
markers_path = None      # đường dẫn file markers

# --------------- Utils ---------------

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

def draw_fps(img, fps, dict_name, frame_markers, collected,
             target, locked, auto_mode):
    if auto_mode and not locked:
        dtxt = "AUTO"
    else:
        dtxt = dict_name + (" (locked)" if locked else "")
    txt = f"FPS:{fps:.1f} dict:{dtxt} frame:{frame_markers} collected:{collected}/{target}"
    cv2.putText(img, txt, (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 180), 2)

def create_map_image(final_ids, show_numbers=True, recording=True):
    """
    Vẽ map: START + 8 ô trên + 8 ô dưới
    Dãy trên (trừ START): fill từ trái sang phải
    Dãy dưới: fill từ phải sang trái
    Canvas ~ 900x600 => vừa nửa trái màn hình.
    """
    h, w = 600, 900
    img = np.full((h, w, 3), 255, dtype=np.uint8)

    sq = 60          # kích thước ô
    thick = 4
    y_top = 100
    y_bottom = 400

    # Ô START bên trái
    start_x = 40
    cv2.rectangle(img, (start_x, y_top),
                  (start_x + sq, y_top + sq),
                  (0, 0, 0), thick)
    cv2.putText(img, "START",
                (start_x + 5, y_top + int(sq * 0.65)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # --- TÍNH VỊ TRÍ HÀNG TRÊN & DƯỚI ĐỂ 8 Ô LUÔN VỪA ---
    num = 8
    margin_left = 160
    margin_right = 40

    avail = w - margin_left - margin_right
    gap = int((avail - num * sq) / (num - 1))
    gap = max(gap, 10)

    # HÀNG TRÊN
    top_slots = []
    for i in range(num):
        x1 = margin_left + i * (sq + gap)
        y1 = y_top
        x2 = x1 + sq
        y2 = y1 + sq
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), thick)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        top_slots.append((cx, cy))

    # HÀNG DƯỚI
    bottom_slots = []
    for i in range(num):
        x1 = margin_left + i * (sq + gap)
        y1 = y_bottom
        x2 = x1 + sq
        y2 = y1 + sq
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), thick)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        bottom_slots.append((cx, cy))

    # Ghi số marker
    if show_numbers:
        positions = top_slots + bottom_slots[::-1]
        for idx, mid in enumerate(final_ids[:len(positions)]):
            cx, cy = positions[idx]
            text = str(mid)
            cv2.putText(img, text, (cx - 20, cy + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 0, 255), 2)

    # Vẽ nút STOP REC / REC OFF
    btn_w, btn_h = 140, 40
    x1_btn = w - btn_w - 30
    y1_btn = 20
    x2_btn = x1_btn + btn_w
    y2_btn = y1_btn + btn_h

    if recording:
        color = (0, 0, 255)   # đỏ
        label = "STOP REC"
    else:
        color = (150, 150, 150)
        label = "REC OFF"

    cv2.rectangle(img, (x1_btn, y1_btn), (x2_btn, y2_btn), color, 2)
    cv2.putText(img, label, (x1_btn + 8, y1_btn + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    global BTN_X1, BTN_Y1, BTN_X2, BTN_Y2
    BTN_X1, BTN_Y1, BTN_X2, BTN_Y2 = x1_btn, y1_btn, x2_btn, y2_btn

    return img

def show_start_screen(window_name="UAVCup Scanner"):
    """Hiện giao diện START, bấm SPACE để bắt đầu scan+record, q để thoát."""
    base = create_map_image([], show_numbers=False, recording=False)
    cv2.putText(base, "Press SPACE to start scanning & recording",
                (140, 260),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(base, "Press 'q' to quit",
                (330, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)

    while True:
        cv2.imshow(window_name, base)
        key = cv2.waitKey(10) & 0xFF
        if key == ord(' '):  # space
            break
        if key == ord('q'):
            cv2.destroyAllWindows()
            sys.exit(0)

# Mouse callback cho nút STOP REC
def on_mouse(event, x, y, flags, param):
    global recording
    if event == cv2.EVENT_LBUTTONDOWN:
        if BTN_X1 <= x <= BTN_X2 and BTN_Y1 <= y <= BTN_Y2:
            if recording:
                recording = False
                print("[INFO] Stop Record clicked -> recording will stop.")

# --------------- MAIN ---------------

def main():
    global recording, video_writer, record_folder, video_path, markers_path

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

    # Target markers (giới hạn max 16 slot)
    target_count = min(args.target, MAX_SLOTS)
    stable_required = args.stable_frames
    id_stability = {}
    final_ids = []
    recent_ids = deque(maxlen=30)

    # FPS state
    frame_markers = 0
    frames = 0
    t0 = time.time()
    fps_val = 0.0

    pyautogui.PAUSE = 0
    pyautogui.FAILSAFE = False

    window_name = "UAVCup Scanner"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(window_name, 0, 0)
    cv2.setMouseCallback(window_name, on_mouse)

    # --- START SCREEN ---
    show_start_screen(window_name)

    # Tạo folder lưu kết quả cho lần chạy này
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    record_folder = os.path.join("records", ts)
    os.makedirs(record_folder, exist_ok=True)
    video_path = os.path.join(record_folder, "scan.mp4")
    markers_path = os.path.join(record_folder, "markers.txt")
    print(f"[INFO] Recording folder: {record_folder}")

    recording = True          # bắt đầu ghi ngay sau khi SPACE
    video_writer = None       # sẽ tạo sau khi có frame đầu tiên

    print("[INFO] Screen ArUco scanner running… Press 'q' to quit.")

    done = False

    while True:
        # Screenshot ROI (right half)
        img = pyautogui.screenshot(region=(roi_left, roi_top, roi_width, roi_height))
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Resize cho processing + recording (video cùng kích cỡ frame này)
        if args.width > 0:
            h, w = frame.shape[:2]
            if w != args.width:
                new_h = int(h * (args.width / float(w)))
                frame = cv2.resize(frame, (args.width, new_h),
                                   interpolation=cv2.INTER_LINEAR)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- Recording ---
        if recording:
            if video_writer is None:
                fh, fw = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(video_path, fourcc, 15.0, (fw, fh))
                print("[INFO] Video recording started:", video_path)
            video_writer.write(frame)
        else:
            if video_writer is not None:
                video_writer.release()
                video_writer = None
                print("[INFO] Video recording stopped.")

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
        else:
            det = detectors[active_dict_name]
            corners, ids, rejected = det.detectMarkers(gray)

        frame_markers = 0
        current_ids = set()

        if ids is not None and len(ids) > 0:
            frame_markers = len(ids)
            current_ids = set(int(m) for m in ids.flatten())

            # Stability logic
            for mid in current_ids:
                id_stability[mid] = id_stability.get(mid, 0) + 1
                if id_stability[mid] == stable_required and mid not in final_ids:
                    if len(final_ids) < MAX_SLOTS:
                        final_ids.append(mid)
                        recent_ids.append(mid)
                        print(f"[INFO] Confirmed marker {mid}. "
                              f"Progress: {len(final_ids)}/{target_count}")

            for mid in list(id_stability.keys()):
                if mid not in current_ids:
                    id_stability[mid] = 0

            if use_pose:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, args.marker, K, D
                )
                # có thể log yaw ra console nếu cần

            # Dừng khi đủ marker
            if len(final_ids) >= target_count:
                done = True
                print("[INFO] Collected required markers.")
                print("RESULT IDs:", final_ids)
                break

        # FPS update
        frames += 1
        if frames % 10 == 0:
            now = time.time()
            elapsed = now - t0
            if elapsed > 0:
                fps_val = 10.0 / elapsed
            t0 = now

        # --- MAP UI ---
        map_img = create_map_image(final_ids, show_numbers=True, recording=recording)
        draw_fps(map_img, fps_val, active_dict_name, frame_markers,
                 len(final_ids), target_count, locked, auto_mode)

        cv2.imshow(window_name, map_img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Đảm bảo đóng writer
    if video_writer is not None:
        video_writer.release()
        video_writer = None

    # Lưu markers ra file
    if markers_path is not None:
        try:
            with open(markers_path, "w", encoding="utf-8") as f:
                f.write(",".join(str(i) for i in final_ids))
            print("[INFO] Saved markers to:", markers_path)
        except Exception as e:
            print("[WARN] Could not save markers:", e)

    if video_path is not None and os.path.exists(video_path):
        print("[INFO] Video file:", video_path)

    # Hiển thị kết quả cuối cùng cho tới khi nhấn 'q'
    final_map = create_map_image(final_ids, show_numbers=True, recording=False)
    if done:
        cv2.putText(final_map, "DONE: all markers collected",
                    (220, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 150, 0), 2)

    print("[INFO] Scan stopped. Press 'q' in the window to close.")
    while True:
        cv2.imshow(window_name, final_map)
        key = cv2.waitKey(50) & 0xFF
        if key == ord('q') or key == 27:
            break

    cv2.destroyAllWindows()
    print("[INFO] Final IDs:", final_ids)


if __name__ == "__main__":
    if not hasattr(cv2, "aruco"):
        print("[ERROR] OpenCV build missing 'aruco'. Install: pip install opencv-contrib-python")
        sys.exit(1)
    main()
