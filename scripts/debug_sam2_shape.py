import cv2
import torch
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

SAM2_CHECKPOINT = "/home/lim/test_ws/checkpoint.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
predictor = SAM2ImagePredictor(sam2_model)

image = np.zeros((360, 640, 3), dtype=np.uint8)
predictor.set_image(image)

boxes = np.array([[10, 10, 50, 50], [60, 60, 100, 100], [110, 110, 150, 150]])
masks, scores, logits = predictor.predict(box=boxes, multimask_output=False)

print("boxes shape:", boxes.shape)
print("masks shape:", masks.shape)
print("scores shape:", scores.shape)
