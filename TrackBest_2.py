#!/usr/bin/env python3
"""
track_3_fixed.py

Strict single-player tracker (movement-first with strong color & feature filtering)
- YOLO (ultralytics) for player detection
- Manual or automatic initialization
- Matching:
    1) Hard color rejection (LAB+HSV histogram distance)
    2) Smallest movement deviation
    3) Direction deviation
    4) Feature deviation
- Will not switch to different players when locked
- Draws rectangle around tracked player and exports annotated video(s)
"""

import os
import sys
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

# Tracking thresholds (tune for your footage)
MAX_MOVE_PER_FRAME = 25
MAX_LOST_FRAMES = 120
COLOR_DISTANCE_THRESHOLD = 0.2
BOX_SCALE = 0.82

MODE = "Manual"       # "Manual" or "Automatic"
TARGET_ID = 1
TARGET_JERSEY = ""

USE_GPU_FOR_YOLO = True

# ---------------- DEVICE & YOLO ----------------
device_yolo = "cuda" if (torch.cuda.is_available() and USE_GPU_FOR_YOLO) else "cpu"
print(f"[INFO] YOLO device: {device_yolo}")

yolo_weights = YOLO_PREFERRED if os.path.exists(YOLO_PREFERRED) else YOLO_FALLBACK
if not os.path.exists(yolo_weights):
    raise FileNotFoundError(f"YOLO weights not found: place {YOLO_PREFERRED} or {YOLO_FALLBACK} in working dir")

print(f"[INFO] Loading YOLO weights: {yolo_weights}")
yolo = YOLO(yolo_weights)
try:
    yolo.to(device_yolo)
except Exception:
    pass

# ---------------- UTILITIES ----------------
def scale_box(xyxy, scale=BOX_SCALE):
    x1,y1,x2,y2 = xyxy
    w,h = x2-x1, y2-y1
    cx,cy = x1+w/2.0, y1+h/2.0
    nw, nh = w*scale, h*scale
    return [float(max(0, cx-nw/2)), float(max(0, cy-nh/2)),
            float(cx+nw/2), float(cy+nh/2)]

def clamp_box(xyxy, W, H):
    x1,y1,x2,y2 = map(int, xyxy)
    x1, x2 = max(0, min(x1, W-1)), max(0, min(x2, W-1))
    y1, y2 = max(0, min(y1, H-1)), max(0, min(y2, H-1))
    if x2<=x1 or y2<=y1: return None
    return [x1,y1,x2,y2]

def crop_from_box(frame, box):
    Hf, Wf = frame.shape[:2]
    cl = clamp_box(box, Wf, Hf)
    if cl is None: return None
    x1,y1,x2,y2 = cl
    return frame[y1:y2, x1:x2]

def bbox_center(box):
    x1,y1,x2,y2 = box
    return np.array([(x1+x2)/2.0, (y1+y2)/2.0], dtype=np.float32)

def get_color_features(crop):
    if crop is None or crop.size==0: return None
    try:
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        lab_hist = cv2.calcHist([lab],[0,1,2],None,[8,8,8],[0,256,0,256,0,256])
        cv2.normalize(lab_hist, lab_hist)

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hsv_hist = cv2.calcHist([hsv],[0,1,2],None,[8,8,8],[0,180,0,256,0,256])
        cv2.normalize(hsv_hist, hsv_hist)

        feat = np.concatenate([lab_hist.flatten(), hsv_hist.flatten()]).astype(np.float32)
        norm = np.linalg.norm(feat)
        if norm>0: feat/=norm
        return feat
    except:
        return None

def color_distance(featA, featB):
    if featA is None or featB is None: return 1.0
    try:
        corr = cv2.compareHist(featA.astype(np.float32), featB.astype(np.float32), cv2.HISTCMP_CORREL)
        return float(max(0.0, 1.0-corr))
    except:
        return 1.0

def ocr_jersey_number(crop):
    if crop is None or crop.size==0: return None
    try:
        h,w = crop.shape[:2]
        if h<30 or w<30: return None
        torso = crop[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
        gray = cv2.cvtColor(torso, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray,(150,75),interpolation=cv2.INTER_CUBIC)
        blurred = cv2.GaussianBlur(gray,(3,3),0)
        sharp = cv2.addWeighted(gray,1.5,blurred,-0.5,0)
        _, thr = cv2.threshold(sharp,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        cfg = "--psm 8 -c tessedit_char_whitelist=0123456789"
        txt = pytesseract.image_to_string(thr, config=cfg, timeout=3).strip()
        return txt if txt.isdigit() else None
    except:
        return None

# ---------------- TRACKER ----------------
class StrictLockedTracker:
    def __init__(self, color_threshold=COLOR_DISTANCE_THRESHOLD, color_feat=None):
        self.locked = False
        self.box = None
        self.center = None
        self.prev_center = None
        self.direction = np.array([0.0,0.0], dtype=np.float32)
        self.color_feat = color_feat
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
        print(f"[INFO] Tracker initialized at {self.center}, jersey={self.jersey}")

    def predict_box(self):
        if self.center is None: return None
        return self.center + self.direction

    def update(self, candidates, max_move=MAX_MOVE_PER_FRAME):
        if not self.locked: return False
        last_center = self.center.copy()

        if not candidates:
            self.lost_count+=1
            pred = self.predict_box()
            if pred is not None and self.box is not None:
                self.prev_center = self.center.copy()
                self.center = pred
                w,h = self.box[2]-self.box[0], self.box[3]-self.box[1]
                cx,cy = self.center
                self.box = [cx-w/2, cy-h/2, cx+w/2, cy+h/2]
            return False

        # Stage1: color filter
        color_filtered=[]
        for c in candidates:
            cf = c.get('color_feat')
            d = color_distance(self.color_feat, cf)
            if cf is not None and d<=self.color_threshold:
                color_filtered.append((c,d))

        if len(color_filtered)==0:
            # relax: pick minimal color distance within movement limit
            best_relaxed=None
            best_score=None
            for c in candidates:
                c_center = bbox_center(c['bbox'])
                move_dev = float(np.linalg.norm(c_center-last_center))
                if move_dev>max_move: continue
                cf = c.get('color_feat')
                d = color_distance(self.color_feat, cf)
                t=(d,move_dev)
                if best_score is None or t<best_score:
                    best_score=t
                    best_relaxed=(c,d)
            if best_relaxed is not None:
                color_filtered.append(best_relaxed)

        if len(color_filtered)==0:
            self.lost_count+=1
            pred=self.predict_box()
            if pred is not None and self.box is not None:
                self.prev_center=self.center.copy()
                self.center=pred
                w,h=self.box[2]-self.box[0], self.box[3]-self.box[1]
                cx,cy=self.center
                self.box=[cx-w/2,cy-h/2,cx+w/2,cy+h/2]
            return False

        # Stage2: movement + direction
        best_candidate=None
        best_tuple=None
        for (c,color_d) in color_filtered:
            c_center=bbox_center(c['bbox'])
            move_dev=float(np.linalg.norm(c_center-last_center))
            if move_dev>max_move: continue
            new_vec=c_center-last_center
            if np.linalg.norm(self.direction)>1e-4 and np.linalg.norm(new_vec)>1e-4:
                cos_sim=np.dot(self.direction,new_vec)/(np.linalg.norm(self.direction)*np.linalg.norm(new_vec))
                cos_sim=float(max(-1.0,min(1.0,cos_sim)))
                dir_dev=1.0-cos_sim
            else: dir_dev=1.0
            t=(move_dev,dir_dev,color_d)
            if best_tuple is None or t<best_tuple:
                best_tuple=t
                best_candidate=c

        if best_candidate is not None:
            self.prev_center=self.center.copy()
            self.center=bbox_center(best_candidate['bbox'])
            self.direction=self.center-self.prev_center
            self.box=[float(x) for x in best_candidate['bbox']]
            cf = best_candidate.get('color_feat')
            if cf is not None:
                self.color_feat=cf
            if self.jersey is None and best_candidate.get('jersey') is not None:
                self.jersey=best_candidate.get('jersey')
            self.lost_count=0
            return True

        self.lost_count+=1
        pred=self.predict_box()
        if pred is not None and self.box is not None:
            self.prev_center=self.center.copy()
            self.center=pred
            w,h=self.box[2]-self.box[0], self.box[3]-self.box[1]
            cx,cy=self.center
            self.box=[cx-w/2,cy-h/2,cx+w/2,cy+h/2]
        return False

    def is_lost(self):
        return self.lost_count>MAX_LOST_FRAMES

# ---------------- MAIN PROCESS ----------------
def process_video(input_path:str, output_path:str):
    cap=cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {input_path}")
    fps=cap.get(cv2.CAP_PROP_FPS) or 30.0
    W,H=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"[INFO] Video: {input_path} ({W}x{H}) fps={fps} frames={total}")

    fourcc=cv2.VideoWriter_fourcc(*"mp4v")
    out_full=cv2.VideoWriter(output_path,fourcc,fps,(W,H))

    frame_idx=0
    color_feat0=None
    tracker=StrictLockedTracker(color_threshold=COLOR_DISTANCE_THRESHOLD, color_feat=color_feat0)
    pbar=tqdm(total=total if total>0 else None, desc="Processing")

    while True:
        ret,frame=cap.read()
        if not ret: break
        frame_idx+=1

        # YOLO detect players
        try:
            res=yolo(frame, imgsz=640, conf=PLAYER_CONF, classes=[0])
        except:
            res=yolo(frame, imgsz=640, conf=PLAYER_CONF, device="cpu", classes=[0])

        p_boxes=res[0].boxes.xyxy.cpu().numpy() if len(res)>0 and getattr(res[0],"boxes",None) else np.array([])

        candidates=[]
        for box in p_boxes:
            scaled=scale_box(box.tolist(), BOX_SCALE)
            crop=crop_from_box(frame, scaled)
            if crop is None: continue
            color_feat=get_color_features(crop)
            jersey=ocr_jersey_number(crop)
            candidates.append({'bbox':scaled,'crop':crop,'color_feat':color_feat,'jersey':jersey})

        # INIT tracker
        if not tracker.locked and len(candidates)>0:
            chosen=None
            if MODE=="Manual":
                idx=max(0,min(len(candidates)-1,TARGET_ID-1))
                chosen=candidates[idx]
            else:
                if TARGET_JERSEY:
                    for c in candidates:
                        if c.get('jersey')==TARGET_JERSEY:
                            chosen=c; break
                if chosen is None:
                    central_x=W/2.0
                    centers_x=[(c['bbox'][0]+c['bbox'][2])/2.0 for c in candidates]
                    idx=int(np.argmin([abs(cx-central_x) for cx in centers_x]))
                    chosen=candidates[idx]
            tracker.initialize(chosen)

        # Update tracker
        tracker.update(candidates, max_move=MAX_MOVE_PER_FRAME)

        # Draw tracker
        if tracker.box is not None and tracker.lost_count==0:
            x1,y1,x2,y2=map(int, tracker.box)
            label="TARGET"+(f" #{tracker.jersey}" if tracker.jersey else "")
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),3)
            cv2.putText(frame,label,(x1,max(12,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)

        if tracker.is_lost():
            cv2.putText(frame, "Target Lost", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0,0,255),2)
            
        out_full.write(frame)
        if pbar: pbar.update(1)

    if pbar: pbar.close()
    cap.release()
    out_full.release()
    print("[DONE] Saved:", output_path)

# ---------------- ENTRY POINT ----------------
if __name__=="__main__":
    if not os.path.exists(INPUT_VIDEO):
        print(f"Input video not found: {INPUT_VIDEO}")
        sys.exit(1)
    print("[INFO] Starting strict tracking. MODE =", MODE)
    process_video(INPUT_VIDEO, OUTPUT_VIDEO)
