import fitz  # PyMuPDF #convert pdf to image 
import torch 
import torchvision  #provide pretrainned model computer vision 
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F   #ransforming images to tensors
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import easyocr   #ocr tool
import numpy as np
import cv2
import io    # handling memory bits 


reader = easyocr.Reader(['en'])         # Load EasyOCR

# Load Categories

with open("resume_coco.json") as f:
    coco = json.load(f)
cat_id_to_name = {c['id']: c['name'] for c in coco['categories']}
num_classes = len(cat_id_to_name) + 1

# Load Model

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = get_model(num_classes)
model.load_state_dict(torch.load("fasterrcnn_resumeAdvance.pth", map_location=device))
model.to(device)
model.eval()   #evaluation mode

# OCR helpers

def crop_and_ocr(image, bbox):
    """Crop a bbox from image and run OCR"""
    x1, y1, x2, y2 = map(int, bbox)
    cropped_img = image.crop((x1, y1, x2, y2))
    cropped_img_np = np.array(cropped_img)

    # Preprocess for OCR
    gray = cv2.cvtColor(cropped_img_np, cv2.COLOR_RGB2GRAY)
    _, threshed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Run OCR
    result = reader.readtext(threshed)
    extracted_text = " ".join([t[1] for t in result]).replace("\n", " ").strip()
    return extracted_text

def run_prediction(pil_img):
    """Run detection + OCR on a PIL image"""
    img_tensor = F.to_tensor(pil_img).to(device)
    orig_w, orig_h = pil_img.size

    with torch.no_grad():
        outputs = model([img_tensor])

    output = outputs[0]
    boxes, labels, scores = output['boxes'], output['labels'], output['scores']

    resume_data = {}
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(pil_img)

    for box, label, score in zip(boxes, labels, scores):
        if score < 0.7:
            continue
        x1, y1, x2, y2 = box.cpu().numpy()

        # OCR
        text = crop_and_ocr(pil_img, [x1, y1, x2, y2])
        resume_data[cat_id_to_name[label.item()]] = text

        # Draw box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f"{cat_id_to_name[label.item()]}: {score:.2f}",
                color='yellow', fontsize=10, backgroundcolor="black")           #give the score for all boxes 

    plt.axis('off')
    plt.show()

    return resume_data

# PDF → Image → Predict → OCR

pdf_path = "./resume/ai-product-manager-resume-example.pdf"
doc = fitz.open(pdf_path)

all_resume_data = {}

for i, page in enumerate(doc):      #loop run for check all pdf page with index 
    pix = page.get_pixmap(dpi=300)    #convert page to image 300plx
    img_bytes = pix.tobytes("png")        
    pil_img = Image.open(io.BytesIO(img_bytes))

    print(f"\n Running prediction + OCR on Page {i+1} ...")
    page_data = run_prediction(pil_img)
    all_resume_data[f"page_{i+1}"] = page_data

# image_path = "Modern-Resume-Free-Template.jpg"   
# pil_img = Image.open(image_path).convert("RGB")

# all_resume_data = {}
# page_data = run_prediction(pil_img)
# all_resume_data["page_1"] = page_data

# Save structured output

output_json = "resume_output.json"
with open(output_json, "w") as f:
    json.dump(all_resume_data, f, indent=4)

print(f"\n Structured resume data saved to {output_json}")
