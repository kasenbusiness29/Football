#!/usr/bin/env python3
"""
track_strict_full_fixed.py

Strict single-player tracker (movement-first with strong color filtering)
- YOLO (ultralytics) for player & ball detection
- Manual or automatic initialization
- Matching:
    1) Hard color rejection (LAB+HSV histogram distance)
    2) smallest movement deviation (distance from last center, with hard threshold)
    3) smallest direction deviation (angle difference vs previous motion)
- Will not switch to distant/different-colored players when locked
- Draws rectangle around tracked player and exports annotated video(s)
"""

import os
import sys
import time
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import pytesseract
from tqdm import tqdm

# ---------------- CONFIG ----------------
INPUT_VIDEO     = "video_2.mp4"
OUTPUT_VIDEO    = "tracked_test.mp4"
HIGHLIGHT_VIDEO = "highlights_test.mp4"

YOLO_PREFERRED = "yolov9n.pt"
YOLO_FALLBACK  = "yolov8x.pt"

PLAYER_CONF = 0.40
BALL_CONF   = 0.45

# Tracking thresholds (tune for your footage)
MAX_MOVE_PER_FRAME = 25   # px - reject candidates further than this from last center
MAX_LOST_FRAMES = 120       # frames before releasing lock
COLOR_DISTANCE_THRESHOLD = 0.2   # 0..1 (lower = more similar). Candidates with larger distance are rejected
DIRECTION_WEIGHT = 1.0
BOX_SCALE = 0.82            # shrink box for crop to focus torso
HIGHLIGHT_DIST_PX = 120

MODE = "Manual"             # "Manual" or "Automatic"
TARGET_ID = 1               # 1-based index for Manual selection
TARGET_JERSEY = "58"          # jersey string for automatic selection (optional)

USE_GPU_FOR_YOLO = True
# Optional: set this to your tesseract executable path if necessary:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

color_feat0 = None
jersey_color_feat0 = None
# ---------------- DEVICE & YOLO ----------------
device_yolo = "cuda" if (torch.cuda.is_available() and USE_GPU_FOR_YOLO) else "cpu"
print(f"[INFO] YOLO device: {device_yolo}")

yolo_weights = YOLO_PREFERRED if os.path.exists(YOLO_PREFERRED) else YOLO_FALLBACK
if not os.path.exists(yolo_weights):
    raise FileNotFoundError(f"YOLO weights not found: try placing {YOLO_PREFERRED} or {YOLO_FALLBACK} in working dir")

print(f"[INFO] Loading YOLO weights: {yolo_weights}")
yolo = YOLO(yolo_weights)
try:
    yolo.to(device_yolo)
except Exception:
    pass

# ---------------- UTILITIES ----------------
def scale_box(xyxy, scale=BOX_SCALE):
    x1,y1,x2,y2 = xyxy
    w,h = (x2-x1), (y2-y1)
    cx,cy = x1 + w/2.0, y1 + h/2.0
    nw, nh = w * scale, h * scale
    nx1 = cx - nw/2.0; ny1 = cy - nh/2.0; nx2 = cx + nw/2.0; ny2 = cy + nh/2.0
    return [float(max(0, nx1)), float(max(0, ny1)), float(max(0, nx2)), float(max(0, ny2))]

def clamp_box(xyxy, W, H):
    x1,y1,x2,y2 = map(int, xyxy)
    x1 = max(0, min(x1, W-1)); x2 = max(0, min(x2, W-1))
    y1 = max(0, min(y1, H-1)); y2 = max(0, min(y2, H-1))
    if x2 <= x1 or y2 <= y1: return None
    return [x1, y1, x2, y2]

def crop_from_box(frame, box):
    Hf, Wf = frame.shape[:2]
    cl = clamp_box(box, Wf, Hf)
    if cl is None: return None
    x1,y1,x2,y2 = cl
    return frame[y1:y2, x1:x2]

def bbox_center(box):
    x1,y1,x2,y2 = box
    return np.array([(x1 + x2)/2.0, (y1 + y2)/2.0], dtype=np.float32)

def bbox_area(box):
    x1,y1,x2,y2 = box
    return max(1.0, (x2-x1)*(y2-y1))

def extract_jersey_mask(crop, ref_hsv_mean, hsv_tolerance=(15, 40, 40)):
    """Binary mask of pixels close to reference jersey color."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lower = np.clip(ref_hsv_mean - hsv_tolerance, [0,0,0], [179,255,255])
    upper = np.clip(ref_hsv_mean + hsv_tolerance, [0,0,0], [179,255,255])
    mask = cv2.inRange(hsv, lower, upper)
    return mask

def get_jersey_color_features(crop, ref_hsv_mean=None):
    """Return jersey color descriptor using masked region only."""
    if crop is None or crop.size == 0:
        return None
    h, w = crop.shape[:2]
    torso = crop[int(h*0.2):int(h*0.55), int(w*0.2):int(w*0.8)]
    if torso.size == 0:
        return None

    if ref_hsv_mean is not None:
        mask = extract_jersey_mask(torso, ref_hsv_mean)
        torso = cv2.bitwise_and(torso, torso, mask=mask)

    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    hsv_mean = np.mean(hsv.reshape(-1,3), axis=0)

    # Histogram
    hist = cv2.calcHist([hsv],[0,1],None,[16,16],[0,180,0,256])
    cv2.normalize(hist, hist)
    feat = np.concatenate([hist.flatten(), hsv_mean]).astype(np.float32)
    feat /= (np.linalg.norm(feat)+1e-6)
    return feat#, hsv_mean

def jersey_color_distance(fA, fB):
    if fA is None or fB is None:
        return 1.0
    return float(1.0 - np.dot(fA, fB))

# Combined color features (LAB + HSV histograms)
def get_color_features(crop):
    """
    Returns a concatenated, normalized LAB(8x8x8) + HSV(8x8x8) histogram vector.
    """
    if crop is None or crop.size == 0:
        return None
    try:
        # LAB
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        lab_hist = cv2.calcHist([lab], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
        cv2.normalize(lab_hist, lab_hist)

        # HSV (note: H range 0..180 in OpenCV)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hsv_hist = cv2.calcHist([hsv], [0,1,2], None, [8,8,8], [0,180,0,256,0,256])
        cv2.normalize(hsv_hist, hsv_hist)

        feat = np.concatenate([lab_hist.flatten(), hsv_hist.flatten()]).astype(np.float32)
        # normalize final vector to unit L2 to make compareHist-compat less sensitive
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat /= norm
        return feat
    except Exception:
        return None

def color_distance(featA, featB):
    """
    Convert correlation to a 0..1 distance. If either is None -> return 1.0 (max distance).
    """
    if featA is None or featB is None:
        return 1.0
    try:
        # Use correlation by cv2.compareHist — requires 1D hist vectors
        # cv2.compareHist expects single-channel histograms; feeding concatenated vectors is fine
        corr = cv2.compareHist(featA.astype(np.float32), featB.astype(np.float32), cv2.HISTCMP_CORREL)
        # correlation in [-1,1], convert to 0..1 distance (lower = more similar)
        return float(max(0.0, 1.0 - corr))
    except Exception:
        # fallback to L2 normalized distance
        try:
            d = np.linalg.norm(featA - featB)
            # scale into 0..1 using a reasonable upper bound (sqrt(2) for two normalized vectors)
            return float(min(1.0, d / np.sqrt(2.0)))
        except Exception:
            return 1.0

def ocr_jersey_number(crop):
    """
    Try to OCR a jersey number from the torso area. Returns string of digits or None.
    """
    if crop is None or crop.size == 0: return None
    try:
        h,w = crop.shape[:2]
        if h < 30 or w < 30: return None
        torso = crop[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
        gray = cv2.cvtColor(torso, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (150,75), interpolation=cv2.INTER_CUBIC)
        blurred = cv2.GaussianBlur(gray, (3,3), 0)
        sharp = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
        _, thr = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cfg = "--psm 8 -c tessedit_char_whitelist=0123456789"
        txt = pytesseract.image_to_string(thr, config=cfg, timeout=3).strip()
        return txt if txt.isdigit() else None
    except Exception:
        return None

# ---------------- TRACKER ----------------
class StrictLockedTracker:
    """
    Strict tracker that prefers color-consistent candidates and will not switch to different-colored players.
    """
    def __init__(self, color_threshold=COLOR_DISTANCE_THRESHOLD, color_feat = None):
        self.locked = False
        self.box = None               # [x1,y1,x2,y2]
        self.center = None
        self.prev_center = None
        self.direction = np.array([0.0,0.0], dtype=np.float32)
        self.color_feat = color_feat        # LAB+HSV concatenated histogram
        self.jersey = None
        self.lost_count = 0
        self.color_threshold = float(color_threshold)

    def initialize(self, detection):
        self.locked = True
        self.box = detection['bbox'].copy()
        self.center = bbox_center(self.box)
        self.prev_center = self.center.copy()
        self.direction = np.array([0.0,0.0], dtype=np.float32)
        self.color_feat = detection.get('color_feat')
        self.jersey = detection.get('jersey')
        self.lost_count = 0
        print(f"[INFO] Tracker initialized at {self.center}, jersey={self.jersey}, color_threshold={self.color_threshold:.3f}")

    def predict_box(self):
        if self.center is None:
            return None
        return self.center + self.direction

    def update(self, candidates, max_move=MAX_MOVE_PER_FRAME):
        """
        candidates: list of dicts with keys 'bbox', 'crop', 'color_feat', 'jersey'
        Returns True if matched & updated, False otherwise
        """
        #if self.color_feat is None:
        #    self.color_feat = color_feat0

        if not self.locked:
            return False
        

        # If no candidates, count lost and predict shift
        if not candidates:
            self.lost_count += 1
            pred = self.predict_box()
            if pred is not None and self.box is not None:
                self.prev_center = self.center.copy()
                self.center = pred
                w = self.box[2] - self.box[0]; h = self.box[3] - self.box[1]
                cx, cy = self.center
                self.box = [cx - w/2.0, cy - h/2.0, cx + w/2.0, cy + h/2.0]
            return False

        last_center = self.center.copy()
        # Stage 1: color filter - only consider candidates whose color distance is <= threshold
        color_filtered = []
        for c in candidates:
            cf = c.get('color_feat')
            d = color_distance(self.color_feat, cf)
            # If the tracker has no color model yet, accept (but we should usually have one after init)
            if self.color_feat is None:
                color_filtered.append((c, d))
            else:
                if d <= self.color_threshold:
                    color_filtered.append((c, d))
                else:
                    # large color dev -> reject candidate
                    pass

        # If no candidate passed color filter, relax slightly: allow one with minimal color distance only if still reasonably close movement-wise
        if len(color_filtered) == 0:
            # Find global best by color among candidates but enforce movement threshold
            best_relaxed = None
            best_relaxed_score = None
            for c in candidates:
                c_center = bbox_center(c['bbox'])
                move_dev = float(np.linalg.norm(c_center - last_center))
                if move_dev > max_move:
                    continue
                cf = c.get('color_feat')
                d = color_distance(self.color_feat, cf)
                # pick the absolute best color even if above threshold, but only if move_dev small
                # create tuple (d, move_dev) - lexicographic
                t = (d, move_dev)
                if best_relaxed_score is None or t < best_relaxed_score:
                    best_relaxed_score = t
                    best_relaxed = (c, d)
            if best_relaxed is not None:
                # only accept relaxed candidate if color distance is not astronomically large
                if best_relaxed[1] <= min(0.6, self.color_threshold + 0.25):
                    color_filtered.append(best_relaxed)

        if len(color_filtered) == 0:
            # nothing acceptable by color -> treat as no match
            self.lost_count += 1
            pred = self.predict_box()
            if pred is not None and self.box is not None:
                self.prev_center = self.center.copy()
                self.center = pred
                w = self.box[2] - self.box[0]; h = self.box[3] - self.box[1]
                cx, cy = self.center
                self.box = [cx - w/2.0, cy - h/2.0, cx + w/2.0, cy + h/2.0]
            return False

        # Stage 2: among color-filtered candidates, pick by movement then direction
        best_candidate = None
        best_tuple = None
        for (c, color_d) in color_filtered:
            c_center = bbox_center(c['bbox'])
            move_dev = float(np.linalg.norm(c_center - last_center))
            if move_dev > max_move:
                continue  # safety
            # direction deviation
            new_vec = c_center - last_center
            if np.linalg.norm(self.direction) > 1e-4 and np.linalg.norm(new_vec) > 1e-4:
                cos_sim = np.dot(self.direction, new_vec) / (np.linalg.norm(self.direction) * np.linalg.norm(new_vec))
                cos_sim = float(max(-1.0, min(1.0, cos_sim)))
                dir_dev = 1.0 - cos_sim
            else:
                dir_dev = 1.0
            # lexicographic tuple: movement -> direction -> color
            t = (move_dev, dir_dev, color_d)
            if best_tuple is None or t < best_tuple:
                best_tuple = t
                best_candidate = c

        if best_candidate is not None:
            # Accept candidate
            self.prev_center = self.center.copy()
            self.center = bbox_center(best_candidate['bbox'])
            self.direction = self.center - self.prev_center
            self.box = [float(x) for x in best_candidate['bbox']]
            # update color model conservatively: only update if color distance small
            cf = best_candidate.get('color_feat')
            if cf is not None:
                if color_distance(self.color_feat, cf) <= (self.color_threshold * 1.0 + 0.05) or self.color_feat is None:
                    # slight running average to adapt slowly
                    if self.color_feat is None:
                        self.color_feat = cf
                    else:
                        self.color_feat = 0.85 * self.color_feat + 0.15 * cf
                        # renormalize
                        nrm = np.linalg.norm(self.color_feat)
                        if nrm > 0:
                            self.color_feat = self.color_feat / nrm
            if self.jersey is None and best_candidate.get('jersey') is not None:
                self.jersey = best_candidate.get('jersey')
            self.lost_count = 0
            return True

        # nothing acceptable
        self.lost_count += 1
        pred = self.predict_box()
        if pred is not None and self.box is not None:
            self.prev_center = self.center.copy()
            self.center = pred
            w = self.box[2] - self.box[0]; h = self.box[3] - self.box[1]
            cx, cy = self.center
            self.box = [cx - w/2.0, cy - h/2.0, cx + w/2.0, cy + h/2.0]
        return False

    def is_lost(self):
        return self.lost_count > MAX_LOST_FRAMES

# ---------------- MAIN PROCESSING ----------------
def process_video(input_path: str, output_path: str, highlight_path: str,
                  mode: str = "Manual", target_id: int = 1, target_jersey: str = ""):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"[INFO] Video: {input_path} ({W}x{H}) fps={fps} frames={total}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_full = cv2.VideoWriter(output_path, fourcc, fps, (W,H))
    out_high = cv2.VideoWriter(highlight_path, fourcc, fps, (W,H))

    pbar = tqdm(total=total if total>0 else None, desc="Processing")
    frame_idx = 0


    # If manual, create image with candidate IDs for first frame
    if mode == "Manual":
        cap2 = cv2.VideoCapture(input_path)
        ret, frame0 = cap2.read()
        cap2.release()
        if not ret:
            print("[WARN] Cannot read first frame for manual selection.")
        else:
            try:
                pres = yolo(frame0, imgsz=640, conf=0.25, classes=[0])
            except Exception:
                pres = yolo(frame0, imgsz=640, conf=0.25, device="cpu", classes=[0])
            p_boxes0 = pres[0].boxes.xyxy.cpu().numpy() if len(pres)>0 and getattr(pres[0], "boxes", None) else np.array([])
            print(f"[INFO] Found {len(p_boxes0)} player detections on first frame.")
            for i, b in enumerate(p_boxes0, start=1):
                x1,y1,x2,y2 = map(int, b)
                if i == TARGET_ID:
                    crop0 = crop_from_box(frame0, b)
                    color_feat0 = get_color_features(crop0)
                    jersey_color_feat0 = get_jersey_color_features(crop0)
                cv2.rectangle(frame0, (x1,y1), (x2,y2), (255,255,255), 2)
                cv2.putText(frame0, f"ID:{i}", (x1, max(0,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.imwrite("first_frame_candidates_strict_fixed.png", frame0)
            print("[INFO] Saved first_frame_candidates_strict_fixed.png -> choose TARGET_ID if using Manual mode.")

    tracker = StrictLockedTracker(color_threshold=COLOR_DISTANCE_THRESHOLD, color_feat=color_feat0)

    # Main loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # YOLO detect players (class 0)
        try:
            p_res = yolo(frame, imgsz=640, conf=PLAYER_CONF, classes=[0])
        except Exception:
            p_res = yolo(frame, imgsz=640, conf=PLAYER_CONF, device="cpu", classes=[0])
        p_boxes = []
        if len(p_res) > 0 and getattr(p_res[0], "boxes", None) is not None:
            p_boxes = p_res[0].boxes.xyxy.cpu().numpy()

        # YOLO detect ball (class 32) for highlight
        try:
            b_res = yolo(frame, imgsz=640, conf=BALL_CONF, classes=[32])
        except Exception:
            b_res = yolo(frame, imgsz=640, conf=BALL_CONF, device="cpu", classes=[32])
        b_boxes = []
        if len(b_res) > 0 and getattr(b_res[0], "boxes", None) is not None:
            b_boxes = b_res[0].boxes.xyxy.cpu().numpy()

        # Build candidates list with scaled bbox, color features, jersey OCR
        candidates = []
        for box in p_boxes:
            scaled = scale_box(box.tolist(), BOX_SCALE)
            crop = crop_from_box(frame, scaled)
            if crop is None: continue
            color_feat = get_color_features(crop)
            jersey_color_feat = get_jersey_color_features(crop)
            if jersey_color_feat0 is not None:
                d_jersey = jersey_color_distance(jersey_color_feat0, jersey_color_feat)
                #print("jersey_color_feat_d")
                #print(d_jersey)
                if d_jersey > 0.1:  
                    # too different → skip this player (likely other team)
                    continue
            #d = color_distance(color_feat0, color_feat)
            #if abs(d) > COLOR_DISTANCE_THRESHOLD: continue
            jersey = ocr_jersey_number(crop)
            candidates.append({'bbox': scaled, 'crop': crop, 'color_feat': color_feat, 'jersey': jersey})
        #print("candidates len")
        #print(len(candidates))
        #break

        # INIT if not locked
        if not tracker.locked:
            if len(candidates) == 0:
                out_full.write(frame)
                if pbar: pbar.update(1)
                continue
            chosen = None
            if mode == "Manual":
                idx = max(0, min(len(candidates)-1, target_id-1))
                chosen = candidates[idx]
            else:
                if target_jersey:
                    for c in candidates:
                        if c.get('jersey') == target_jersey:
                            chosen = c; break
                if chosen is None:
                    centers_x = [(c['bbox'][0] + c['bbox'][2]) / 2.0 for c in candidates]
                    central_x = W/2.0
                    idx = int(np.argmin([abs(cx - central_x) for cx in centers_x]))
                    chosen = candidates[idx]
            tracker.initialize(chosen)
            # draw initial box
            if tracker.box is not None:
                x1,y1,x2,y2 = map(int, tracker.box)
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 3)
                label = "TARGET" + (f" #{tracker.jersey}" if tracker.jersey else "")
                cv2.putText(frame, label, (x1, max(12,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            out_full.write(frame)
            if pbar: pbar.update(1)
            continue

        # Update tracker with candidates
        tracker.update(candidates, max_move=MAX_MOVE_PER_FRAME)

        # If lost for too long, reset to allow re-init
        if tracker.is_lost():
            print(f"[WARN] Lost target for {tracker.lost_count} frames -> clearing lock.")
            cv2.putText(frame, "lost", (200, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            tracker = StrictLockedTracker(color_threshold=COLOR_DISTANCE_THRESHOLD, color_feat=color_feat0)
            out_full.write(frame)
            if pbar: pbar.update(1)
            continue

        # Draw locked box (green) and label
        if tracker.box is not None and tracker.lost_count == 0:
            x1,y1,x2,y2 = map(int, tracker.box)
            label = "TARGET"
            if tracker.jersey:
                label += f" #{tracker.jersey}"
            # draw filled shadow for readability
            #cv2.rectangle(frame, (x1-2,y1-24), (x1+220,y1), (0,0,0), -1)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 3)
            cv2.putText(frame, label, (x1, max(12,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        # Ball draw & highlight decision
        is_highlight = False
        if isinstance(b_boxes, np.ndarray) and b_boxes.shape[0] > 0 and tracker.box is not None and tracker.lost_count == 0:
            # pick largest ball candidate
            areas = [(b[2]-b[0])*(b[3]-b[1]) for b in b_boxes]
            idx = int(np.argmax(areas))
            bb = b_boxes[idx]
            ball_pos = (int((bb[0]+bb[2])/2), int((bb[1]+bb[3])/2))
            cv2.circle(frame, ball_pos, 8, (0,0,255), -1)
            px = int((tracker.box[0] + tracker.box[2]) / 2.0)
            py = int((tracker.box[1] + tracker.box[3]) / 2.0)
            if np.linalg.norm(np.array([px,py]) - np.array(ball_pos)) < HIGHLIGHT_DIST_PX:
                is_highlight = True

        out_full.write(frame)
        if is_highlight:
            out_high.write(frame)

        if pbar: pbar.update(1)

    # cleanup
    if pbar: pbar.close()
    cap.release()
    out_full.release()
    out_high.release()
    print("[DONE] Saved:", output_path, highlight_path)

# ---------------- ENTRY POINT ----------------
if __name__ == "__main__":
    if not os.path.exists(INPUT_VIDEO):
        print(f"Input video not found: {INPUT_VIDEO}")
        sys.exit(1)
    print("[INFO] Starting strict tracking (fixed). MODE =", MODE)
    process_video(INPUT_VIDEO, OUTPUT_VIDEO, HIGHLIGHT_VIDEO, mode=MODE, target_id=TARGET_ID, target_jersey=TARGET_JERSEY)
