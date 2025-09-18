import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json

with open("resume_coco.json") as f:
    coco = json.load(f)
cat_id_to_name = {c['id']: c['name'] for c in coco['categories']}
num_classes = len(cat_id_to_name) + 1  # +1 for background

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = get_model(num_classes)
model.load_state_dict(torch.load("fasterrcnn_resume.pth", map_location=device))
model.to(device)
model.eval()


image_path = "3.png"  
img = Image.open(image_path).convert("RGB")
img_tensor = F.to_tensor(img).to(device)


with torch.no_grad():
    outputs = model([img_tensor])

output = outputs[0]  
boxes = output['boxes']
labels = output['labels']
scores = output['scores']


fig, ax = plt.subplots(1, figsize=(12,12))
ax.imshow(img)


for box, label, score in zip(boxes, labels, scores):
    if score < 0.5:
        continue
    x1, y1, x2, y2 = box.cpu().numpy()
    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.text(x1, y1-5, f"{cat_id_to_name[label.item()]} {score:.2f}", color='yellow', fontsize=10, backgroundcolor="black")

plt.show()


from PIL import Image
import numpy as np
import cv2
import json
import math
import easyocr

# ---------------------------
# CONFIG
# ---------------------------
IMAGE_PATH = "3.png"  
BOXES_FORMAT = "xyxy"  

DET_CANVAS_W, DET_CANVAS_H = None, None   

pred_boxes = [
    {"bbox": [50, 40, 400, 100], "label": "name"},
    {"bbox": [50, 110, 400, 160], "label": "personal_info"},
    {"bbox": [50, 170, 400, 230], "label": "skills"},
    {"bbox": [50, 240, 400, 350], "label": "work_experience"},
    {"bbox": [50, 360, 400, 410], "label": "certificates"},
    {"bbox": [50, 420, 400, 460], "label": "hobby"},
]

def to_xyxy(box, fmt):
    if fmt == "xyxy":
        x1, y1, x2, y2 = box
    else:
        cx, cy, w, h = box
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2
    return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]

def rescale_box(x1, y1, x2, y2, orig_w, orig_h, det_w=None, det_h=None):
    """If det_w/h provided, map detector coords -> original image coords."""
    if det_w and det_h:
        sx = orig_w / float(det_w)
        sy = orig_h / float(det_h)
        x1 = int(round(x1 * sx)); x2 = int(round(x2 * sx))
        y1 = int(round(y1 * sy)); y2 = int(round(y2 * sy))
    # clamp
    x1 = max(0, min(x1, orig_w-1)); x2 = max(0, min(x2, orig_w))
    y1 = max(0, min(y1, orig_h-1)); y2 = max(0, min(y2, orig_h))
    if x2 <= x1: x2 = min(orig_w, x1+5)
    if y2 <= y1: y2 = min(orig_h, y1+5)
    return x1, y1, x2, y2

def preprocess_roi(gray):
    den = cv2.bilateralFilter(gray, 7, 50, 50)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    con = clahe.apply(den)

    up = cv2.resize(con, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    _, th = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # small opening to remove dots/bullets noise (keep text strokes)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

    return up, opened

def run_easyocr(reader, img_gray):
    up, th = preprocess_roi(img_gray)

    # Try original
    res1 = reader.readtext(up, detail=1, paragraph=True, rotation_info=[0,90,180,270])

    # Try inverted (helps white text on dark background)
    inv = cv2.bitwise_not(up)
    res2 = reader.readtext(inv, detail=1, paragraph=True, rotation_info=[0,90,180,270])

    def score(results):
        if not results: return 0.0
        confs = [r[2] for r in results if len(r) >= 3]
        return np.mean(confs) if confs else 0.0

    pick = res1 if score(res1) >= score(res2) else res2
    text = " ".join([r[1] for r in pick if len(r) >= 2])
    return text

def clean_text(t):
    # Common OCR fixes
    t = t.replace("_", " ")
    t = t.replace("  ", " ")
    # 10096, 0096 → 100%, 0%
    t = t.replace("096", "%")
    t = t.replace("0 96", "%")
    # bullets become odd chars sometimes
    t = t.replace("•", "-").replace("●", "-").replace("", "-")
    return t.strip()

# ---------------------------
# MAIN
# ---------------------------

reader = easyocr.Reader(['en'], gpu=False)
pil_img = Image.open(IMAGE_PATH).convert("RGB")
orig_w, orig_h = pil_img.size
cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
gray_full = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

resume_data = {}

for det in pred_boxes:
    x1, y1, x2, y2 = to_xyxy(det['bbox'], BOXES_FORMAT)
    x1, y1, x2, y2 = rescale_box(x1, y1, x2, y2, orig_w, orig_h, DET_CANVAS_W, DET_CANVAS_H)

    roi_gray = gray_full[y1:y2, x1:x2]
    if roi_gray.size == 0:
        resume_data[det['label']] = ""
        continue

    text = run_easyocr(reader, roi_gray)
    resume_data[det['label']] = clean_text(text)

# Save + print
with open("resume_output.json", "w", encoding="utf-8") as f:
    json.dump(resume_data, f, indent=4, ensure_ascii=False)

print(json.dumps(resume_data, indent=4, ensure_ascii=False))
