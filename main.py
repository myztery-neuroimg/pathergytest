#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pathergy Test Longitudinal Alignment + Composite Timeline
Description:
  Aligns all pathergy test images to the Day-1 baseline forearm contour,
  detects papules, overlays bounding boxes, and creates a labeled
  side-by-side montage showing morphological evolution over time.
"""

import cv2, numpy as np, math
from PIL import Image, ImageDraw, ImageFont

# --- Input paths ---
im1_path = "day1_0h.png"      # Day-1 baseline at test
im2_path = "day1_24h.png"          
im3_path = "day2_48h.png"           

# --- Load images ---
img1 = Image.open(im1_path).convert("RGB")
img2 = Image.open(im2_path).convert("RGB")
img3 = Image.open(im3_path).convert("RGB")
base_size = img1.size  # (width, height)

# ---------- Helper functions ----------
def affine_register(src_pil, dst_pil):
    """Estimate affine transform aligning src → dst using SIFT + RANSAC."""
    src = cv2.cvtColor(np.array(src_pil), cv2.COLOR_RGB2GRAY)
    dst = cv2.cvtColor(np.array(dst_pil), cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    k1,d1 = sift.detectAndCompute(src,None)
    k2,d2 = sift.detectAndCompute(dst,None)
    bf = cv2.BFMatcher(cv2.NORM_L2,crossCheck=True)
    m = sorted(bf.match(d1,d2), key=lambda x:x.distance)[:60]
    src_pts = np.float32([k1[x.queryIdx].pt for x in m]).reshape(-1,1,2)
    dst_pts = np.float32([k2[x.trainIdx].pt for x in m]).reshape(-1,1,2)
    M,_ = cv2.estimateAffinePartial2D(src_pts,dst_pts,method=cv2.RANSAC,ransacReprojThreshold=4)
    return M

def detect_papules_red(pil_img,min_area=30,max_cnt=2):
    bgr = cv2.cvtColor(np.array(pil_img),cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,(0,60,60),(12,255,255)) | cv2.inRange(hsv,(170,60,60),(180,255,255))
    mask = cv2.medianBlur(mask,3)
    contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    pts=[]
    for c in contours:
        a=cv2.contourArea(c)
        if a>=min_area:
            M=cv2.moments(c)
            if M["m00"]:
                cx=int(M["m10"]/M["m00"]); cy=int(M["m01"]/M["m00"])
                pts.append((cx,cy,a))
    pts.sort(key=lambda t:t[2],reverse=True)
    return [p[:2] for p in pts[:max_cnt]]

def detect_papules_dark(pil_img,scan_region=None):
    gray=cv2.cvtColor(np.array(pil_img),cv2.COLOR_RGB2GRAY)
    H,W=gray.shape
    if not scan_region:
        x0,y0=int(W*0.25),int(H*0.45)
        x1,y1=int(W*0.80),int(H*0.85)
    else:
        x0,y0,x1,y1=scan_region
    crop=gray[y0:y1,x0:x1]
    clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    g=clahe.apply(crop)
    _,thr=cv2.threshold(g,int(np.mean(g)-10),255,cv2.THRESH_BINARY_INV)
    thr=cv2.medianBlur(thr,3)
    cnts,_=cv2.findContours(thr,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    blobs=[]
    for c in cnts:
        a=cv2.contourArea(c)
        if 18<=a<=420:
            x,y,w,h=cv2.boundingRect(c)
            cx,cy=x+w//2,y+h//2
            per=cv2.arcLength(c,True)
            circ=(4*math.pi*a)/(per*per+1e-6)
            blobs.append((cx,cy,a,circ))
    best=None;bestscore=1e9
    for i in range(len(blobs)):
        for j in range(i+1,len(blobs)):
            d=math.hypot(blobs[i][0]-blobs[j][0],blobs[i][1]-blobs[j][1])
            if 15<=d<=65:
                score=d-5*(blobs[i][3]+blobs[j][3])
                if score<bestscore:
                    bestscore=score;best=(blobs[i],blobs[j])
    if not best:return []
    b1,b2=best
    return [(x0+int(b1[0]),y0+int(b1[1])),(x0+int(b2[0]),y0+int(b2[1]))]

def transform_points(pts,M):
    out=[]
    for (x,y) in pts:
        v=np.dot(M,np.array([x,y,1]))
        out.append((int(v[0]),int(v[1])))
    return out

def warp_to_base(src_pil,M,size):
    src=cv2.cvtColor(np.array(src_pil),cv2.COLOR_RGB2BGR)
    warped=cv2.warpAffine(src,M,size,flags=cv2.INTER_LINEAR)
    return Image.fromarray(cv2.cvtColor(warped,cv2.COLOR_BGR2RGB))

def draw_boxes(pil_img,pts,label):
    out=pil_img.copy()
    d=ImageDraw.Draw(out)
    for i,(x,y) in enumerate(pts[:2],start=1):
        r=22
        d.rectangle([x-r,y-r,x+r,y+r],outline=(255,0,0),width=4)
        d.text((x+r+6,y-10),f"P{i}",fill=(255,0,0))
    d.text((10,10),label,fill=(255,255,255))
    return out

# ---------- Detect papules ----------
p1_pts = detect_papules_red(img1)
p2_pts = detect_papules_red(img2)
p3_pts = detect_papules_dark(img3)

# ---------- Register to baseline ----------
M_2to1 = affine_register(img2,img1)
M_3to1 = affine_register(img3,img1)
W1,H1 = base_size
img2_w = warp_to_base(img2,M_2to1,(W1,H1))
img3_w = warp_to_base(img3,M_3to1,(W1,H1))
p2_base = transform_points(p2_pts,M_2to1)
p3_base = transform_points(p3_pts,M_3to1)

# ---------- Overlays ----------
out1 = draw_boxes(img1,p1_pts,"Day (0h)")
out2 = draw_boxes(img2_w,p2_base,"Day 1 (~24h)")
out3 = draw_boxes(img3_w,p3_base,"Day 2 (~48h)")

# ---------- Build composite panel ----------
pad = 20
montage_width = W1*3 + pad*2
montage_height = H1
montage = Image.new("RGB",(montage_width,montage_height),(0,0,0))
montage.paste(out1,(0,0))
montage.paste(out2,(W1+pad,0))
montage.paste(out3,(2*(W1+pad)-pad,0))

# Optional caption bar
d = ImageDraw.Draw(montage)
caption = "Pathergy Test Timeline (Baseline-Aligned to Day 1)"
d.text((10,H1-30),caption,fill=(255,255,255))

montage.save("pathergy_timeline_composite.jpg",quality=95)
print("Composite saved as pathergy_timeline_composite.jpg")

