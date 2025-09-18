import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import easyocr
import numpy as np
import cv2
import re

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Load your COCO categories
with open("resume_coco.json") as f:
    coco = json.load(f)
cat_id_to_name = {c['id']: c['name'] for c in coco['categories']}
num_classes = len(cat_id_to_name) + 1

# Get FasterRCNN model
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_model(num_classes)
model.load_state_dict(torch.load("fasterrcnn_resume.pth", map_location=device))
model.to(device)
model.eval()

# --------------------------
# Utility Functions
# --------------------------

def rescale_box(x1, y1, x2, y2, orig_w, orig_h, det_w, det_h):
    sx, sy = orig_w / det_w, orig_h / det_h
    return [int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)]

def crop_and_ocr(image, bbox, orig_w, orig_h, det_w, det_h):
    x1, y1, x2, y2 = rescale_box(*bbox, orig_w, orig_h, det_w, det_h)
    cropped_img = image.crop((x1, y1, x2, y2))
    cropped_img_np = np.array(cropped_img)

    gray = cv2.cvtColor(cropped_img_np, cv2.COLOR_BGR2GRAY)
    _, threshed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    result = reader.readtext(threshed)
    text = " ".join([t[1] for t in result]).replace("\n", " ").strip()
    return text

def clean_text(label, text):
    text = text.strip()

    if label == "personal information":
        # Fix phone, email, LinkedIn
        email = re.search(r'[\w\.-]+@[\w\.-]+', text)
        phone = re.search(r'(\+?\d[\d\s-]{8,15})', text)
        linkedin = re.search(r'(linkedin\.com\S*)', text, re.IGNORECASE)

        return {
            "email": email.group(0) if email else "",
            "phone": phone.group(0) if phone else "",
            "linkedin": linkedin.group(0) if linkedin else "",
            "raw": text
        }

    if label == "skills":
        skills = [s.strip().capitalize() for s in text.replace(",", " ").split() if len(s) > 2]
        return skills

    if label == "education":
        # Extract degree + institution + years
        degree = re.search(r'(Bachelor|Master|B\.Sc|M\.Sc|Ph\.D|Diploma).*', text, re.IGNORECASE)
        years = re.findall(r'(\d{4})', text)
        return {
            "degree": degree.group(0) if degree else "",
            "years": f"{years[0]} - {years[1]}" if len(years) >= 2 else "",
            "raw": text
        }

    if label == "name":
        return text.replace("WORK EXPERIENCE", "").strip()

    return text

# --------------------------
# Process Image
# --------------------------
image_path = "./JPEGImages/29.PNG"
img = Image.open(image_path).convert("RGB")
img_tensor = F.to_tensor(img).to(device)

orig_w, orig_h = img.size
det_w, det_h = img_tensor.shape[2], img_tensor.shape[1]

with torch.no_grad():
    outputs = model([img_tensor])

output = outputs[0]
boxes, labels, scores = output['boxes'], output['labels'], output['scores']

structured_resume = {
    "name": "",
    "personal_info": {},
    "education": {},
    "skills": [],
    "work_experience": ""
}

image_copy = img.copy()
ax = plt.gca()

for box, label, score in zip(boxes, labels, scores):
    if score < 0.5:
        continue

    x1, y1, x2, y2 = box.cpu().numpy()
    label_name = cat_id_to_name[label.item()]
    text = crop_and_ocr(img, [x1, y1, x2, y2], orig_w, orig_h, det_w, det_h)
    cleaned = clean_text(label_name, text)

    if label_name == "name":
        structured_resume["name"] = cleaned
    elif label_name == "personal information":
        structured_resume["personal_info"] = cleaned
    elif label_name == "education":
        structured_resume["education"] = cleaned
    elif label_name == "skills":
        structured_resume["skills"] = cleaned
    elif label_name == "work experince":
        structured_resume["work_experience"] = cleaned

    # Draw box
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                             linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.text(x1, y1 - 5, f"{label_name}: {score:.2f}",
            color='yellow', fontsize=10, backgroundcolor="black")

plt.imshow(image_copy)
plt.axis('off')
plt.show()

# Save structured JSON
with open("resume_output2.json", "w") as f:
    json.dump(structured_resume, f, indent=4)

print("Structured Resume JSON:")
print(json.dumps(structured_resume, indent=4))
