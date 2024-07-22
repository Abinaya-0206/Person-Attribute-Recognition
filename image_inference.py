import argparse
import os

import numpy as np
import torch
from PIL import Image, ImageFont, ImageDraw
import torchvision.transforms as transforms
import cv2

from configs import cfg, update_config
from models.base_block import FeatClassifier
from models.backbone import swin_transformer, resnet, bninception
from models.model_factory import build_backbone, build_classifier
from models.backbone.tresnet import tresnet
from tools.function import get_model_log_path, get_reload_weight
from tools.utils import set_seed, str2bool
from ultralytics import YOLO  

set_seed(605)

def main(cfg, args):
    attribute_names = [
        'accessoryHat', 'accessoryMuffler', 'accessoryNothing', 'accessorySunglasses', 'hairLong',
        'upperBodyCasual', 'upperBodyFormal', 'upperBodyJacket', 'upperBodyLogo', 'upperBodyPlaid',
        'upperBodyShortSleeve', 'upperBodyThinStripes', 'upperBodyTshirt', 'upperBodyOther', 'upperBodyVNeck',
        'lowerBodyCasual', 'lowerBodyFormal', 'lowerBodyJeans', 'lowerBodyShorts', 'lowerBodyShortSkirt', 'lowerBodyTrousers',
        'footwearLeatherShoes', 'footwearSandals', 'footwearShoes', 'footwearSneaker',
        'carryingBackpack', 'carryingOther', 'carryingMessengerBag', 'carryingNothing', 'carryingPlasticBags',
        'personalLess30', 'personalLess45', 'personalLess60', 'personalLarger60',
        'personalMale'
    ]

    attribute_short_names = {
        'accessoryHat': 'Hat',
        'accessoryMuffler': 'Muffler',
        'accessoryNothing': 'NoAccessory',
        'accessorySunglasses': 'Sunglasses',
        'hairLong': 'LongHair',
        'upperBodyCasual': 'CasualUpper',
        'upperBodyFormal': 'FormalUpper',
        'upperBodyJacket': 'Jacket',
        'upperBodyLogo': 'Logo',
        'upperBodyPlaid': 'Plaid',
        'upperBodyShortSleeve': 'ShortSleeve',
        'upperBodyThinStripes': 'ThinStripes',
        'upperBodyTshirt': 'Tshirt',
        'upperBodyOther': 'OtherUpper',
        'upperBodyVNeck': 'VNeck',
        'lowerBodyCasual': 'CasualLower',
        'lowerBodyFormal': 'FormalLower',
        'lowerBodyJeans': 'Jeans',
        'lowerBodyShorts': 'Shorts',
        'lowerBodyShortSkirt': 'ShortSkirt',
        'lowerBodyTrousers': 'Trousers',
        'footwearLeatherShoes': 'LeatherShoes',
        'footwearSandals': 'Sandals',
        'footwearShoes': 'Shoes',
        'footwearSneaker': 'Sneakers',
        'carryingBackpack': 'Backpack',
        'carryingOther': 'OtherCarrying',
        'carryingMessengerBag': 'MessengerBag',
        'carryingNothing': 'NoCarry',
        'carryingPlasticBags': 'PlasticBags'
    }

    exp_dir = os.path.join('exp_result', cfg.DATASET.NAME)
    model_dir, log_dir = get_model_log_path(exp_dir, cfg.NAME)

    # Load the model architecture
    backbone, c_output = build_backbone(cfg.BACKBONE.TYPE, cfg.BACKBONE.MULTISCALE)
    classifier = build_classifier(cfg.CLASSIFIER.NAME)(
        nattr=len(attribute_names),
        c_in=c_output,
        bn=cfg.CLASSIFIER.BN,
        pool=cfg.CLASSIFIER.POOLING,
        scale=cfg.CLASSIFIER.SCALE
    )
    model = FeatClassifier(backbone, classifier)

    # Load the model weights
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    # Modify the path to your .pth file
    model_path = '/content/drive/MyDrive/Intern/Person Attribute/data/PETA/ckpt_max_2024-05-09_05_11_43.pth'
    model = get_reload_weight(model_dir, model, pth=model_path)
    model.eval()

    # Load YOLOv8 model for person detection
    yolo_model = YOLO('data/PETA/yolov8n.pt')

    # Load and preprocess the image
    image_path = 'data/PETA/a4.png'
    img = Image.open(image_path).convert("RGB")

    # Define preprocess transformation for attribute model
    preprocess = transforms.Compose([
        transforms.Resize((256, 192), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img_draw = img.copy()
    draw = ImageDraw.Draw(img_draw)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

    # Perform person detection using YOLOv8
    results = yolo_model(np.array(img))

    # Define threshold for attribute detection
    threshold = 0.7

    age_indices = [attribute_names.index('personalLess30'), attribute_names.index('personalLess45'),
                   attribute_names.index('personalLess60'), attribute_names.index('personalLarger60')]
    age_labels = ['Less than 30', '30-45', '45-60', '60+']

    gender_index = attribute_names.index('personalMale')

    # Load custom font
    font_path = "data/PETA/Calibri Regular.ttf"
    font_size = 25
    font = ImageFont.truetype(font_path, font_size)

    # Process each detected person
    for i, box in enumerate(results[0].boxes):
        if box.conf < 0.6:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        person_img = img.crop((x1, y1, x2, y2))
        person_tensor = preprocess(person_img)
        person_tensor = person_tensor.unsqueeze(0).cuda()

        # Perform inference
        with torch.no_grad():
            logits, attns = model(person_tensor)
            logits_tensor = logits[0]
            output = torch.sigmoid(logits_tensor)

        probabilities = output.cpu().numpy().flatten()  # Convert tensor to numpy array and flatten

        if len(probabilities) < max(age_indices) + 1:
            print(f"Skipping due to insufficient probability length: {len(probabilities)}")
            continue

        # Separate attributes into age, gender, and other attributes
        age_attribute = age_labels[np.argmax([probabilities[i] for i in age_indices])]
        gender = "Male" if probabilities[gender_index] > threshold else "Female"
        other_attributes = [attribute_short_names[attribute_names[j]] for j in range(len(attribute_names)) if j not in age_indices + [gender_index] and probabilities[j] > threshold]

        # Draw bounding box and attributes
        color = colors[i % len(colors)]
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)

        age_text = f"Age: {age_attribute}"
        gender_text = f"Gender: {gender}"
        other_text = [f"{attr}" for attr in other_attributes]

        combined_text = f"{gender_text}, {age_text}\n"

        for j in range(0, len(other_text), 2):
            combined_text += ', '.join(other_text[j:j+2]) + "\n"

        # Determine text position
        text_width, text_height = draw.textsize(combined_text, font=font)
        padding = 5
        text_x = x1
        text_y = y1 - text_height - padding if y1 - text_height - padding > 0 else y2 + padding

        # Adjust text position if it goes out of frame
        if text_x + text_width > img.width:
            text_x = img.width - text_width - padding
        if text_y + text_height > img.height:
            text_y = img.height - text_height - padding

        # Create a new image with an alpha layer (RGBA) for drawing the text box with transparency
        text_box_image = Image.new('RGBA', img_draw.size, (0, 0, 0, 0))
        text_box_draw = ImageDraw.Draw(text_box_image)

        # Draw semi-transparent background with lower alpha for translucency
        background_x1 = text_x - padding / 2
        background_y1 = text_y - padding / 2
        background_x2 = text_x + text_width + padding / 2
        background_y2 = text_y + text_height + padding / 2
        text_box_draw.rectangle([(background_x1, background_y1), (background_x2, background_y2)], fill=(0, 0, 0, 100))

        img_draw = Image.alpha_composite(img_draw.convert('RGBA'), text_box_image)

        draw = ImageDraw.Draw(img_draw)
        draw.text((text_x + padding / 2, text_y + padding / 2), combined_text, fill=color, font=font)

    # Save the annotated image
    output_image_path = 'output/annotated_new8.png'
    img_draw.save(output_image_path)
    print(f"Annotated image saved at {output_image_path}")

def argument_parser():
    parser = argparse.ArgumentParser(description="attribute recognition",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--cfg", help="decide which cfg to use", type=str,
    )
    parser.add_argument("--debug", type=str2bool, default="true")

    args = parser.parse_args()
    update_config(cfg, args)

    return cfg, args

if __name__ == "__main__":
    cfg, args = argument_parser()
    main(cfg, args)
