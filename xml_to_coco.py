import os
import json
import xml.etree.ElementTree as ET

def xml_to_coco(xml_dir, image_dir, output_file):
    images, annotations, categories = [], [], []
    class_map = {}  
    ann_id = 1
    cat_id = 1

    for xml_file in os.listdir(xml_dir):
        if not xml_file.endswith(".xml"):
            continue
        tree = ET.parse(os.path.join(xml_dir, xml_file))
        root = tree.getroot()
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in class_map:
                class_map[class_name] = cat_id
                categories.append({"id": cat_id, "name": class_name})
                cat_id += 1

    for xml_file in os.listdir(xml_dir):
        if not xml_file.endswith(".xml"):
            continue
        tree = ET.parse(os.path.join(xml_dir, xml_file))
        root = tree.getroot()
        filename = root.find('filename').text
        width = int(root.find('size')[0].text)
        height = int(root.find('size')[1].text)

        image_id = int(os.path.splitext(filename)[0].replace("resume","")) # or any unique id
        images.append({"file_name": filename, "height": height, "width": width, "id": image_id})

        for obj in root.findall('object'):
            class_name = obj.find('name').text
            cat_id = class_map[class_name]
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            w = xmax - xmin
            h = ymax - ymin

            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": [xmin, ymin, w, h],
                "area": w*h,
                "iscrowd": 0
            })
            ann_id += 1

    coco = {"images": images, "annotations": annotations, "categories": categories}
    with open(output_file, "w") as f:
        json.dump(coco, f, indent=4)

xml_to_coco("Annotations", "JPEGImages", "resume_coco.json")

