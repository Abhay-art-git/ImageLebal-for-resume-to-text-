import os
import json
import random
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class ResumeDataset(Dataset):
    def __init__(self, images_dir, annotations_file, transforms=None):
        self.images_dir = images_dir
        self.transforms = transforms
        
        
        with open(annotations_file) as f:
            coco = json.load(f)
        
        self.images = {img['id']: img for img in coco['images']}
        self.annotations = {}
        for ann in coco['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)
        
        self.img_ids = list(self.images.keys())
        self.cat_id_to_name = {c['id']: c['name'] for c in coco['categories']}
        self.num_classes = len(coco['categories']) + 1  

    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.images[img_id]
        img_path = os.path.join(self.images_dir, img_info['file_name'])
        img = Image.open(img_path).convert("RGB")
        
        annots = self.annotations.get(img_id, [])
        boxes = []
        labels = []
        for ann in annots:
            boxes.append(ann['bbox'])  
            labels.append(ann['category_id'])
        
        boxes = torch.as_tensor([[x, y, x+w, y+h] for x, y, w, h in boxes], dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([img_id])}
        
        if self.transforms:
            img = self.transforms(img)
        else:
            img = F.to_tensor(img)
        
        return img, target

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

images_dir = "JPEGImages"      
annotations_file = "resume_coco.json" 

torch.manual_seed(42)

dataset = ResumeDataset(images_dir, annotations_file)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = get_model(dataset.num_classes)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for imgs, targets in train_loader:
        imgs = list(img.to(device) for img in imgs)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(imgs, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
    
    lr_scheduler.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "fasterrcnn_resume.pth")
print("Model saved as fasterrcnn_resume.pth")


model.eval()
with torch.no_grad():
    imgs, targets = next(iter(test_loader))
    imgs = list(img.to(device) for img in imgs)
    outputs = model(imgs)
    
    for img_tensor, output in zip(imgs, outputs):
        img_np = img_tensor.permute(1,2,0).cpu().numpy()
        fig, ax = plt.subplots(1)
        ax.imshow(img_np)
        for box, label in zip(output['boxes'], output['labels']):
            x1, y1, x2, y2 = box.cpu().numpy()
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1-5, dataset.cat_id_to_name[label.item()], color='yellow', fontsize=10, backgroundcolor="black")
        plt.show()
