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

reader = easyocr.Reader(['en'])

with open("resume_coco.json") as f:
    coco = json.load(f)
cat_id_to_name = {c['id']: c['name'] for c in coco['categories']}
num_classes = len(cat_id_to_name) + 1  

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

image_path = "./JPEGImages/60.PNG"  
img = Image.open(image_path).convert("RGB")
img_tensor = F.to_tensor(img).to(device)

def rescale_box(x1, y1, x2, y2, orig_w, orig_h, det_w, det_h):
    """
    Rescale bounding box coordinates from detector image to original image size.
    """
    sx, sy = orig_w / det_w, orig_h / det_h
    x1, x2 = x1 * sx, x2 * sx
    y1, y2 = y1 * sy, y2 * sy
    return [int(x1), int(y1), int(x2), int(y2)]

def crop_and_ocr(image, bbox, orig_w, orig_h, det_w, det_h):
    x1, y1, x2, y2 = rescale_box(bbox[0], bbox[1], bbox[2], bbox[3], orig_w, orig_h, det_w, det_h)
    cropped_img = image.crop((x1, y1, x2, y2))
    cropped_img_np = np.array(cropped_img)


    gray = cv2.cvtColor(cropped_img_np, cv2.COLOR_BGR2GRAY)
    _, threshed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

 
    result = reader.readtext(threshed)
    extracted_text = " ".join([t[1] for t in result]).replace("\n", " ").strip()

    return extracted_text


orig_w, orig_h = img.size
det_w, det_h = img_tensor.shape[2], img_tensor.shape[1]  


with torch.no_grad():
    outputs = model([img_tensor])

output = outputs[0]  
boxes = output['boxes']
labels = output['labels']
scores = output['scores']


resume_data = {}


image_copy = img.copy()
draw = ImageDraw.Draw(image_copy)


for box, label, score in zip(boxes, labels, scores):
    if score < 0.5:  
        continue

    x1, y1, x2, y2 = box.cpu().numpy()

  
    text = crop_and_ocr(img, [x1, y1, x2, y2], orig_w, orig_h, det_w, det_h)
    resume_data[cat_id_to_name[label.item()]] = text  

   
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
    ax = plt.gca()
    ax.add_patch(rect)
    ax.text(x1, y1 - 5, f"{cat_id_to_name[label.item()]}: {score:.2f}", color='yellow', fontsize=10, backgroundcolor="black")


plt.imshow(image_copy)
plt.axis('off')
plt.show()


print(resume_data)


output_json = "resume_output.json"
with open(output_json, "w") as f:
    json.dump(resume_data, f, indent=4)

print(f"Structured resume data saved to {output_json}")
