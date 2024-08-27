# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import json
from typing import Any
import numpy as np
import random
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from cog import BasePredictor, Input, Path, BaseModel
from typing import List
from subprocess import call

HOME = os.getcwd()
os.chdir("GroundingDINO")
call("pip install -q .", shell=True)
os.chdir(HOME)
os.chdir("segment_anything")
call("pip install -q .", shell=True)
os.chdir(HOME)

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import (
    clean_state_dict,
    get_phrases_from_posmap,
)

# segment anything
from segment_anything import build_sam, build_sam_hq, SamPredictor

from ram.models import ram


class ModelOutput(BaseModel):
    combinde_masked_img: Path
    binary_masked_imgs: List[Path]


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.image_size = 384
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                normalize,
            ]
        )

        # load model
        self.ram_model = ram(
            pretrained="pretrained/ram_swin_large_14m.pth",
            image_size=self.image_size,
            vit="swin_l",
        )
        self.ram_model.eval()
        self.ram_model = self.ram_model.to(self.device)

        self.model = load_model(
            "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            "pretrained/groundingdino_swint_ogc.pth",
            device=self.device,
        )

        self.sam = SamPredictor(
            build_sam(checkpoint="pretrained/sam_vit_h_4b8939.pth").to(self.device)
        )
        self.sam_hq = SamPredictor(
            build_sam_hq(checkpoint="pretrained/sam_hq_vit_h.pth").to(self.device)
        )

    def predict(
            self,
            input_image: Path = Input(description="Input image"),
            use_sam_hq: bool = Input(
                description="Use sam_hq instead of SAM for prediction", default=False
            ),
    ) -> ModelOutput:
        """Run a single prediction on the model"""

        # default settings
        box_threshold = 0.25
        text_threshold = 0.2
        iou_threshold = 0.5

        image_pil, image_tensor = load_image(str(input_image))

        raw_image = image_tensor.resize((self.image_size, self.image_size))
        raw_image = self.transform(raw_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            tags, tags_chinese = self.ram_model.generate_tag(raw_image)

        tags = tags[0].replace(" |", ",")
        tags = tags.lower()
        tags = tags.strip()
        if not tags.endswith("."):
            tags = tags + "."

        CLASSES = tags.split(",")

        predictor = self.sam_hq if use_sam_hq else self.sam

        image = cv2.imread(str(input_image))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)

        detections = self.model.predict_with_classes(
            image=image,
            classes=enhance_class_name(class_names=CLASSES),
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )

        detections.mask = segment(
            sam_predictor=predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )

        MASK_SAVE_PATH = f"/tmp/individual_masks"
        COMBINED_MASK_SAVE_PATH = f"{MASK_SAVE_PATH}/combined_mask_with_boundaries.png"
        BINARY_MASK_PATHS = []

        # Process and save each detection mask individually
        for i, (detection_mask, xyxy, confidence, class_id) in enumerate(
                zip(detections.mask, detections.xyxy, detections.confidence, detections.class_id)):
            # Ensure the mask is binary (0 or 1)
            binary_mask = detection_mask.astype(np.uint8)

            # Create a visualizable mask (0 or 255)
            # visual_mask = binary_mask * 255

            # Get the class name
            class_name = CLASSES[class_id]

            binary_mask_path = f"{MASK_SAVE_PATH}/binary_mask_{i}_{class_name}.png"
            # Save the binary mask
            cv2.imwrite(binary_mask_path, binary_mask * 255)
            BINARY_MASK_PATHS.append(Path(binary_mask_path))

            # Save the visual mask
            # cv2.imwrite(f"{MASK_SAVE_PATH}/visual_mask_{i}_{class_name}.png", visual_mask)

            print(f"Saved mask for object {i}: class={class_name}, confidence={confidence:.2f}")

        combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for i, detection_mask in enumerate(detections.mask):
            # Dilate the current mask
            kernel = np.ones((3, 3), np.uint8)
            dilated_mask = cv2.dilate(detection_mask.astype(np.uint8), kernel, iterations=1)

            # Subtract the original mask from the dilated mask to get the boundary
            boundary = dilated_mask - detection_mask.astype(np.uint8)

            # Update the combined mask:
            # 1. Add the current mask
            # 2. Subtract the boundary where it overlaps with existing objects
            combined_mask = np.maximum(combined_mask, detection_mask.astype(np.uint8)) - np.minimum(boundary,
                                                                                                    combined_mask)

        # Save the combined mask
        cv2.imwrite(COMBINED_MASK_SAVE_PATH, combined_mask * 255)
        print("Saved combined mask with boundaries between connected objects")

        return ModelOutput(
            combinde_masked_img=Path(COMBINED_MASK_SAVE_PATH),
            binary_masked_imgs=BINARY_MASK_PATHS,
        )


def enhance_class_name(class_names: List[str]) -> List[str]:
    return [
        f"all {class_name}s"
        for class_name
        in class_names
    ]


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(
        clean_state_dict(checkpoint["model"]), strict=False
    )
    print(load_res)
    _ = model.eval()
    return model


def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)
