import os
import cv2
import json
import torch
import numpy as np
import pycocotools.mask as mask_util
from pathlib import Path
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import supervision as sv
from ultralytics import YOLO

# --- 전역 설정 (serm 데이터셋용) ---
# YOLO: serm 데이터셋으로 fine-tuned YOLO11x
YOLO_MODEL_PATH = "/home/mscautoev/test_ws/yolo11/ultralytics/runs/detect/train22/weights/best.pt"

# YOLO 탐지 결과에 대한 최소 신뢰도 임계값
CONFIDENCE_THRESHOLD = 0.5

# SAM2: serm 데이터셋으로 fine-tuned SAM2.1 Base+
SAM2_CHECKPOINT = "/home/mscautoev/checkpoint_sam2_serm.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# YOLO 모델의 클래스 이름 매핑 (serm data.yaml의 names 순서와 일치)
YOLO_CLASS_NAMES = [
    'blueLine', 'whiteLine', 'yellowLine', 'stopLine', 'crossWalk',
    'goAhead', 'turnLeft', 'turnRight', 'aheadOrTurnLeft', 'aheadOrTurnRight',
    'numbers', 'texts', 'slowDown', 'others'
]
ID_TO_CLASS_NAME = {i: name for i, name in enumerate(YOLO_CLASS_NAMES)}

# --- 모델 초기화 (스크립트 실행 시 한 번만 수행) ---
print(f"SAM2 모델 로드 중...")
sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

print(f"YOLO 모델 로드 중: {YOLO_MODEL_PATH}")
yolo_model = YOLO(YOLO_MODEL_PATH)


def process_image_with_yolo_sam2(image_path, output_base_dir, yolo_model, sam2_predictor, confidence_threshold, id_to_class_name_map):
    """
    단일 이미지에 대해 YOLO 탐지, SAM-2 세그멘테이션, 시각화 및 JSON 저장을 수행합니다.

    Args:
        image_path (str): 처리할 이미지 파일의 전체 경로.
        output_base_dir (Path): 결과를 저장할 기본 출력 디렉토리 (하위 폴더는 함수 내에서 생성).
        yolo_model (YOLO): 초기화된 YOLO 모델 객체.
        sam2_predictor (SAM2ImagePredictor): 초기화된 SAM2 예측기 객체.
        confidence_threshold (float): YOLO 탐지에 대한 최소 신뢰도 임계값.
        id_to_class_name_map (dict): 클래스 ID를 이름으로 매핑하는 딕셔너리.
    """
    print(f"\n--- 이미지 처리 시작: {os.path.basename(image_path)} ---")

    # 이미지 로딩 (SAM2 및 시각화용)
    img = cv2.imread(image_path)
    if img is None:
        print(f"오류: 이미지 파일 '{image_path}'을(를) 로드할 수 없습니다. 건너뜜.")
        return
    h, w, _ = img.shape

    # YOLO 추론 수행
    yolo_results = yolo_model(source=image_path, verbose=False)

    # YOLO 탐지 결과 추출 및 필터링
    yolo_detected_boxes = []
    yolo_detected_class_ids = []
    yolo_detected_confidences = []

    for r in yolo_results:
        if r.boxes:
            for box, conf, cls in zip(r.boxes.xyxy.cpu().numpy(),
                                       r.boxes.conf.cpu().numpy(),
                                       r.boxes.cls.cpu().numpy()):
                if conf >= confidence_threshold:
                    yolo_detected_boxes.append(box)
                    yolo_detected_class_ids.append(int(cls))
                    yolo_detected_confidences.append(float(conf))

    yolo_detected_boxes_np = np.array(yolo_detected_boxes)
    yolo_detected_class_ids_np = np.array(yolo_detected_class_ids, dtype=int)
    yolo_detected_confidences_np = np.array(yolo_detected_confidences)

    if yolo_detected_boxes_np.size == 0:
        print(f"  YOLO가 {confidence_threshold} 이상의 신뢰도를 가진 객체를 탐지하지 못했습니다. SAM2 건너뜜.")
        return # 객체가 없으면 SAM2 및 시각화 건너뛰기

    print(f"  YOLO가 필터링 후 총 {len(yolo_detected_boxes_np)}개의 객체를 탐지했습니다 (신뢰도 > {confidence_threshold}).")

    # SAM2 적용
    sam2_predictor.set_image(img)
    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=yolo_detected_boxes_np,
        multimask_output=False,
    )

    # 마스크 차원 조정 (N, 1, H, W) -> (N, H, W)
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    # --- 시각화 ---
    # 파일명 접두사 (원본 이미지 파일명에서 확장자 제외)
    base_stem = Path(image_path).stem 

    # 1. SAM2 마스크만 플롯된 이미지 생성
    sam2_only_detections = sv.Detections(
        xyxy=yolo_detected_boxes_np, 
        mask=masks.astype(bool),
        class_id=yolo_detected_class_ids_np
    )
    mask_annotator_only = sv.MaskAnnotator()
    annotated_frame_masks_only = mask_annotator_only.annotate(scene=img.copy(), detections=sam2_only_detections)
#     output_image_path_masks_only = str(output_base_dir / f"{base_stem}_sam2_masks_only.jpg")
#     cv2.imwrite(output_image_path_masks_only, annotated_frame_masks_only)
#     print(f"  결과 이미지 저장됨 (SAM2 마스크만): {output_image_path_masks_only}")

    # 2. 기존 YOLO bbox + SAM2 마스크 + YOLO 라벨이 모두 플롯된 이미지 생성
    detections_full = sv.Detections(
        xyxy=yolo_detected_boxes_np,
        mask=masks.astype(bool),
        class_id=yolo_detected_class_ids_np,
        confidence=yolo_detected_confidences_np
    )
    labels = [
        f"{id_to_class_name_map.get(class_id, 'Unknown')} {confidence:.2f}"
        for class_id, confidence in zip(detections_full.class_id, detections_full.confidence)
    ]
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    mask_annotator = sv.MaskAnnotator()

    annotated_frame_full = img.copy()
    annotated_frame_full = box_annotator.annotate(scene=img.copy(), detections=detections_full)
    # annotated_frame_full = img.copy()
    # annotated_frame_full = label_annotator.annotate(scene=annotated_frame_full, detections=detections_full, labels=labels)
    annotated_frame_full = mask_annotator.annotate(scene=annotated_frame_full, detections=detections_full)

    output_image_path_full = str(output_base_dir / f"{base_stem}_yolo_sam2_segmentation.jpg")
    cv2.imwrite(output_image_path_full, annotated_frame_full)
    print(f"  결과 이미지 저장됨 (YOLO bbox + SAM2 마스크 + 라벨): {output_image_path_full}")

    # --- JSON 저장 ---
    def single_mask_to_rle(mask):
        rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
        rle["counts"] = rle["counts"].decode("utf-8")
        return rle

    mask_rles = [single_mask_to_rle(mask) for mask in masks]
    # scores가 torch.Tensor 또는 numpy.ndarray 일 때 tolist()를 호출.
    # 단일 float 값일 경우 리스트로 감싸줌.
    if isinstance(scores, torch.Tensor):
        sam2_scores = scores.squeeze().tolist()
    elif isinstance(scores, np.ndarray):
        sam2_scores = scores.squeeze().tolist()
    elif isinstance(scores, float): # scores가 이미 float인 경우
        sam2_scores = [scores] # 리스트로 감싸줌
    else: # 기타 예상치 못한 타입의 경우 (예: int 등)
        sam2_scores = [float(scores)] # float로 변환 후 리스트로 감싸줌

    results_json_data = {
        "image_path": image_path, # 원본 이미지 경로
        "img_width": w,
        "img_height": h,
        "annotations": [
            {
                "class_id": int(class_id),
                "class_name": id_to_class_name_map.get(class_id, 'Unknown'),
                "bbox_xyxy": box.tolist(),
                "bbox_confidence": float(yolo_confidence),
                "segmentation_rle": mask_rle,
                "segmentation_score": float(sam2_score),
            }
            for box, class_id, yolo_confidence, mask_rle, sam2_score in zip(
                yolo_detected_boxes_np,
                yolo_detected_class_ids_np,
                yolo_detected_confidences_np,
                mask_rles,
                sam2_scores
            )
        ],
    }

#     output_json_path = str(output_base_dir / f"{base_stem}_yolo_sam2_result.json")
#     with open(output_json_path, "w") as f:
#         json.dump(results_json_data, f, indent=4)
#     print(f"  JSON 결과 저장됨: {output_json_path}")


# --- 메인 실행 블록 ---
if __name__ == "__main__":
    # ----------------------------------------------------
    # serm 데이터셋 테스트 이미지 경로
    INPUT_IMAGE_BASE_DIR = "/home/mscautoev/test_ws/sam2.1_finetune/data/1_serm_processed/test"
    # 결과 저장 경로
    OUTPUT_BASE_DIR = Path("outputs/yolo_sam2_serm")
    # ----------------------------------------------------

    # 결과 기본 출력 디렉토리 생성
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n'{INPUT_IMAGE_BASE_DIR}' 폴더 및 하위 폴더의 모든 이미지 분석 시작...")

    # 이미지 파일 확장자 목록
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff') # 필요에 따라 추가

    # 입력 폴더 존재 여부 확인
    if not os.path.isdir(INPUT_IMAGE_BASE_DIR):
        print(f"오류: 입력 이미지 기본 폴더 '{INPUT_IMAGE_BASE_DIR}'를 찾을 수 없습니다. 경로를 확인해주세요.")
        exit()

    # os.walk를 사용하여 모든 하위 폴더를 순회
    for root, dirs, files in os.walk(INPUT_IMAGE_BASE_DIR):
        for filename in files:
            if filename.lower().endswith(image_extensions):
                full_image_path = os.path.join(root, filename)
                
                # 결과 이미지를 저장할 서브 폴더 생성 (입력 폴더 구조를 미러링)
                # 입력 기본 디렉토리로부터의 상대 경로 계산
                relative_path_to_image_dir = os.path.relpath(root, INPUT_IMAGE_BASE_DIR)
                current_output_dir = OUTPUT_BASE_DIR / relative_path_to_image_dir
                current_output_dir.mkdir(parents=True, exist_ok=True)

                # 단일 이미지 처리 함수 호출
                process_image_with_yolo_sam2(
                    image_path=full_image_path,
                    output_base_dir=current_output_dir,
                    yolo_model=yolo_model,
                    sam2_predictor=sam2_predictor,
                    confidence_threshold=CONFIDENCE_THRESHOLD,
                    id_to_class_name_map=ID_TO_CLASS_NAME
                )
            else:
                pass # 이미지 파일이 아닌 경우 건너뜜

    print("\n--- 모든 이미지에 대한 YOLO + SAM-2 세그멘테이션 완료 ---")
  