#!/usr/bin/env python3
"""

- Uses Ultralyics YOLO for inference (keeps your OBB support).
- Annotates frames (res = res.plot()) and writes to an output video.
- Logs detections to console and to detections.log (timestamped lines).
- Removes all matplotlib GUI / interactive display code so it runs headless.

Configuration: edit the CONFIG block below or pass environment changes as needed.

Author: adapted from user's original script (converted to headless)
"""

from __future__ import annotations

import time
import cv2
from ultralytics import YOLO
import numpy as np
from datetime import datetime
import os
import logging
import sys

# ------------------ CONFIG ------------------
MODEL_PATH = "best_fp32_2.onnx"   # or "yolov8s.pt" or "best.pt"
VIDEO_DEVICE = 0                  # webcam index (change to 0,1,... as needed)
OUTPUT_VIDEO = "annotated.avi"
CONF_THRESHOLD = 0.25
LOG_FILE = "detections.log"
# -------------------------------------------

# Initialize model (keep OBB task if using rotated-box model)
model = YOLO(MODEL_PATH, task="obb") if MODEL_PATH.endswith(".onnx") else YOLO(MODEL_PATH)

# Setup camera capture
cap = cv2.VideoCapture(VIDEO_DEVICE)
if not cap.isOpened():
    raise IOError(f"Cannot open camera device: {VIDEO_DEVICE}")

# video writer setup
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
fps_input = cap.get(cv2.CAP_PROP_FPS) or 20.0
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps_input, (w, h))

# Setup logging (console + file)
logger = logging.getLogger("headless_yolo")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(message)s")

# console handler
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

# file handler (append)
fh = logging.FileHandler(LOG_FILE, mode="a")
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)

# write header line in log file once (optional)
logger.info(f"Detections log started: {datetime.now().isoformat()}")

# Helper: extraction function (kept mostly as-is)
def extract_detections(result):
    """
    Returns list of detections (dicts) with keys:
    {class_id, class_name, conf, coords}
    Handles both result.boxes and result.obb and common coord attrs.
    """
    dets = []
    boxes_obj = None
    # prefer obb if present (for rotated-box models)
    if getattr(result, "obb", None) is not None and len(getattr(result, "obb")):
        boxes_obj = result.obb
    elif getattr(result, "boxes", None) is not None and len(getattr(result, "boxes")):
        boxes_obj = result.boxes
    else:
        return dets

    # find coordinates attribute
    coords = None
    for attr in ("xyxy", "xywh", "xyxys", "data"):
        coords = getattr(boxes_obj, attr, None)
        if coords is not None:
            break

    # convert to numpy if tensor-like
    try:
        if hasattr(coords, "cpu"):
            coords = coords.cpu().numpy()
        coords = np.array(coords)
    except Exception:
        pass

    cls_array = getattr(boxes_obj, "cls", None)
    conf_array = getattr(boxes_obj, "conf", None)

    # convert class/conf arrays if needed
    if cls_array is not None and hasattr(cls_array, "cpu"):
        try:
            cls_array = cls_array.cpu().numpy()
        except Exception:
            pass
    if conf_array is not None and hasattr(conf_array, "cpu"):
        try:
            conf_array = conf_array.cpu().numpy()
        except Exception:
            pass

    # determine number of detections
    try:
        n = len(coords)
    except Exception:
        n = len(cls_array) if cls_array is not None else 0

    for i in range(n):
        # coords row -> list
        try:
            row = coords[i].tolist() if hasattr(coords[i], "tolist") else list(coords[i])
        except Exception:
            try:
                row = list(coords)
            except Exception:
                row = []

        # class id
        cls_id = None
        if cls_array is not None:
            try:
                cls_id = int(cls_array[i])
            except Exception:
                try:
                    cls_id = int(np.array(cls_array)[i])
                except Exception:
                    cls_id = None

        # confidence
        conf = None
        if conf_array is not None:
            try:
                conf = float(conf_array[i])
            except Exception:
                try:
                    conf = float(np.array(conf_array)[i])
                except Exception:
                    conf = None

        # name lookup (safe guard if model.names missing keys)
        try:
            class_name = model.names[cls_id] if (cls_id is not None and cls_id in model.names) else (str(cls_id) if cls_id is not None else "unknown")
        except Exception:
            # fallback if model.names is not indexable in same way
            try:
                class_name = model.names[int(cls_id)]
            except Exception:
                class_name = str(cls_id) if cls_id is not None else "unknown"

        dets.append({
            "class_id": cls_id,
            "class_name": class_name,
            "conf": conf,
            "coords": row
        })

    return dets

# Main loop (headless)
prev_time = time.time()
frame_count = 0
frame_idx = 0
last_frame_time = time.time()  # for instantaneous FPS

try:
    logger.info("Starting capture loop (headless). Press Ctrl-C to stop.")
    while True:
        success, frame = cap.read()
        if not success:
            logger.info("Camera read failed — exiting loop.")
            break

        # compute instantaneous FPS
        now = time.time()
        frame_time = now - last_frame_time if last_frame_time else 0.0
        instant_fps = 1.0 / frame_time if frame_time > 0 else 0.0
        last_frame_time = now

        # inference
        # pass the frame directly (Ultralytics supports np array input)
        results = model(frame, conf=CONF_THRESHOLD)
        res = results[0]

        # annotated BGR image (plot returns annotated image)
        annotated = res.plot()

        # extract detections and counts
        detections = extract_detections(res)
        class_counts = {}
        for d in detections:
            class_counts[d["class_name"]] = class_counts.get(d["class_name"], 0) + 1

        # FPS calc (averaged, updated every 0.5s)
        frame_count += 1
        now2 = time.time()
        dt = now2 - prev_time
        if dt >= 0.5:
            avg_fps = frame_count / dt
            prev_time = now2
            frame_count = 0
            # log averaged + instantaneous fps (once per averaged update)
            timestamp = datetime.now().isoformat()
            logger.info(f"[{timestamp}] FPS avg:{avg_fps:.1f} fps_instant:{instant_fps:.1f}")
        else:
            avg_fps = None

        # overlay lines: one line per class + FPS (draw on annotated image)
        base_x, base_y = 10, 20
        line_spacing = 22
        overlay_lines = []
        if class_counts:
            for k, v in sorted(class_counts.items(), key=lambda kv: -kv[1]):
                overlay_lines.append(f"{k}: {v}")
        else:
            overlay_lines.append("Detections: 0")

        # show both averaged and instantaneous fps if available
        if avg_fps is not None:
            overlay_lines.append(f"FPS(avg): {avg_fps:.1f}")
        overlay_lines.append(f"FPS(inst): {instant_fps:.1f}")

        for i, line in enumerate(overlay_lines):
            y = base_y + i * line_spacing
            cv2.putText(
                annotated,
                line,
                (base_x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

        # write annotated frame to video
        out.write(annotated)

        # log each detection: console + file via logger
        timestamp = datetime.now().isoformat()
        if detections:
            for d in detections:
                coords_str = ", ".join([f"{float(v):.2f}" if isinstance(v, (int, float, np.number)) else str(v) for v in d["coords"]])
                conf_str = f"{d['conf']:.3f}" if d["conf"] is not None else "N/A"
                log_line = f"[{timestamp}] frame:{frame_idx} class:{d['class_name']} id:{d['class_id']} conf:{conf_str} coords:[{coords_str}]"
                logger.info(log_line)
        else:
            logger.info(f"[{timestamp}] frame:{frame_idx} no_detections")

        frame_idx += 1

except KeyboardInterrupt:
    logger.info("Interrupted by user (KeyboardInterrupt).")
except Exception as e:
    logger.exception("Unexpected error in capture loop: %s", e)
finally:
    # release resources
    try:
        cap.release()
    except Exception:
        pass
    try:
        out.release()
    except Exception:
        pass

    logger.info(f"Saved {OUTPUT_VIDEO}")
    logger.info(f"Detections logged to: {LOG_FILE}")