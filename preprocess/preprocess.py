import torch
import numpy as np
import random
import os

import os
import json
from tqdm import tqdm
import shutil

from glob import glob
from sklearn.model_selection import train_test_split
import yaml
import ast
from IPython.core.magic import register_line_cell_magic


def my_seed_everywhere(seed: int = 42):
    random.seed(seed) # random
    np.random.seed(seed) # numpy
    os.environ["PYTHONHASHSEED"] = str(seed) # os
    # pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 


def convert_bbox_coco2yolo(img_width, img_height, bbox):
    """
    Convert bounding box from COCO  format to YOLO format

    Parameters
    ----------
    img_width : int
        width of image
    img_height : int
        height of image
    bbox : list[int]
        bounding box annotation in COCO format: 
        [top left x position, top left y position, width, height]

    Returns
    -------
    list[float]
        bounding box annotation in YOLO format: 
        [x_center_rel, y_center_rel, width_rel, height_rel]
    """
    
    # YOLO bounding box format: [x_center, y_center, width, height]
    # (float values relative to width and height of image)
    x_tl, y_tl, w, h = bbox

    dw = 1.0 / img_width
    dh = 1.0 / img_height

    x_center = x_tl + w / 2.0
    y_center = y_tl + h / 2.0

    x = x_center * dw
    y = y_center * dh
    w = w * dw
    h = h * dh

    return [x, y, w, h]

def make_folders(path="output"):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    return path


def convert_coco_json_to_yolo_txt(output_path, json_file):

    path = make_folders(output_path)

    with open(json_file, encoding='utf-8') as f:
        json_data = json.load(f)

    # write _darknet.labels, which holds names of all classes (one class per line)
    label_file = os.path.join(output_path, "data.names")
    with open(label_file, "w", encoding='utf-8') as f:
        for category in tqdm(json_data["categories"], desc="Categories"):
            category_name = category["name"]
            f.write(f"{category_name}\n")

    for image in tqdm(json_data["images"], desc="Annotation txt for each iamge"):
        img_id = image["id"]
        img_name = image["file_name"]
        img_width = image["width"]
        img_height = image["height"]

        anno_in_image = [anno for anno in json_data["annotations"] if anno["image_id"] == img_id]
        anno_txt = os.path.join(output_path, img_name.split(".")[0] + ".txt")
        with open(anno_txt, "w", encoding='utf-8') as f:
            for anno in anno_in_image:
                category = anno["category_id"] - 1
                bbox_COCO = anno["bbox"]
                x, y, w, h = convert_bbox_coco2yolo(img_width, img_height, bbox_COCO)
                f.write(f"{category} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    print("Converting COCO Json to YOLO txt finished!")


if __name__ == "__main__":
    print("Preprocessing Started!")
    
    my_seed_everywhere(42)

    # Convert Json to Txt
    #convert_coco_json_to_yolo_txt("../dataset/train-001/labels", "../dataset/train-001/label/Train.json")

    # Prepare Datasets
    img_list = glob('../dataset/train-001-aug-re/images/*.png')
    print("Total Image Number:",len(img_list))
    train_img_list, val_img_list = train_test_split(img_list, test_size=0.05, random_state=42)
    print("Train and Val Image Number:", len(train_img_list), len(val_img_list))

    with open('../dataset/train-001-aug-re/train.txt', 'w', encoding='utf8') as f:
        f.write('\n'.join(train_img_list) + '\n')

    with open('../dataset/train-001-aug-re/val.txt', 'w', encoding='utf8') as f:
        f.write('\n'.join(val_img_list) + '\n')

    data = {}
    data['train'] = '../dataset/train-001-aug-re/train.txt'
    data['val'] = '../dataset/train-001-aug-re/val.txt'
    data['test'] = '../dataset/test-002/images'
    data['nc'] = 14
    data['names'] = ['세단(승용차)', 'SUV', '승합차', '버스', '학원차량(통학버스)', '트럭', '택시', '성인', '어린이', '오토바이', '전동킥보드', '자전거', '유모차', '쇼핑카트']


    with open('../dataset/train-001-aug-re/data.yaml', 'w', encoding='utf8') as f:
        yaml.dump(data, f)

    print(data)

    with open("../dataset/train-001-aug-re/data.yaml", 'r') as stream:
        names = str(yaml.safe_load(stream)['names'])

    namesFile = open("../dataset/train-001-aug-re/data.names", "w+", encoding='utf8')
    names = ast.literal_eval(names)
    for name in names:
        namesFile.write(name +'\n')
    namesFile.close()

    print("Preprocessing Finished!")