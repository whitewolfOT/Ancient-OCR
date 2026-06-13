"""Test CC-based word splitting on 1.jpg line 0 only."""
from __future__ import annotations
import json, sys
from pathlib import Path
import cv2, numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from ocr_engine.kraken_backend import KrakenBackend
from ocr_engine.profile_loader import OCRProfile

seg_data = json.loads(Path("data/debug/1_segmentation.json").read_text())
lines     = seg_data["lines"]
line0     = lines[0]
boundary  = line0["boundary"]

img       = cv2.imread("data/test_images/1.jpg")
bxs = [int(p[0]) for p in boundary]; bys = [int(p[1]) for p in boundary]
lx1 = max(0, min(bxs)); lx2 = min(img.shape[1], max(bxs))
ly1 = max(0, min(bys)); ly2 = min(img.shape[0], max(bys))
line_crop = cv2.cvtColor(img[ly1:ly2, lx1:lx2], cv2.COLOR_BGR2GRAY)

print(f"Line 0 crop: x={lx1}..{lx2}, y={ly1}..{ly2}, shape={line_crop.shape}")

profile = OCRProfile(name="andalusian_naskh")
backend = KrakenBackend(profile=profile)
word_bboxes = backend._words_from_cc(line_crop, lx1, ly1)

print(f"\nCC word detection: {len(word_bboxes)} words (RTL order)")
print(f"{'#':>3}  {'x':>6}  {'y':>5}  {'w':>5}  {'h':>5}  right")
for i, (x, y, w, h) in enumerate(word_bboxes):
    print(f"{i+1:>3}  {x:>6}  {y:>5}  {w:>5}  {h:>5}  {x+w}")
