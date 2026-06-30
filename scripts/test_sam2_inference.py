import os
import cv2
import torch
import numpy as np
import json
from pathlib import Path
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def main():
    # 설정 경로
    SAM2_CHECKPOINT = "/home/lim/test_ws/checkpoint.pt"
    SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    INPUT_DIR = "/home/lim/test_ws/cropped_test_000176"
    OUTPUT_DIR = "/home/lim/test_ws/cropped_test_inference_results"
    
    print(f"SAM2 모델 로드 중 (Config: {SAM2_MODEL_CONFIG}, Device: {DEVICE})...")
    sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
    
    # AutomaticMaskGenerator 대신 명시적 프롬프팅을 위한 Predictor 사용
    predictor = SAM2ImagePredictor(sam2_model)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n추론 및 폴리곤 변환을 시작합니다...")
    for root, dirs, files in os.walk(INPUT_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                rel_path = os.path.relpath(img_path, INPUT_DIR)
                out_path_img = os.path.join(OUTPUT_DIR, rel_path)
                out_path_json = os.path.splitext(out_path_img)[0] + ".json"
                os.makedirs(os.path.dirname(out_path_img), exist_ok=True)
                
                print(f"처리 중: {rel_path}")
                image = cv2.imread(img_path)
                if image is None:
                    continue
                
                h, w = image.shape[:2]
                
                # SAM2는 RGB 입력을 기대함
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                predictor.set_image(image_rgb)
                
                # 크롭된 이미지 자체에 객체가 꽉 차있다고 가정하고, 전체를 감싸는 BBox를 프롬프트로 줌
                box_prompt = np.array([0, 0, w, h])
                
                # 프롬프트를 주어 추론 (크롭 이미지 내에서 단일 객체 추출)
                masks, scores, _ = predictor.predict(
                    box=box_prompt,
                    multimask_output=False
                )
                
                # 가장 점수가 높은 마스크 1개 선택 (multimask_output=False이므로 1개만 반환됨)
                mask = masks[0]
                
                # ---------------------------------------------------------
                # 2. 마스크를 폴리곤(Polygon)으로 변환
                # ---------------------------------------------------------
                mask_uint8 = (mask * 255).astype(np.uint8)
                # 외곽선 추출
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                polygons = []
                # 라벨 이름은 폴더명에서 추출 (예: whiteLine, yellowLine)
                label_name = os.path.basename(os.path.dirname(img_path)) 
                
                for contour in contours:
                    # 너무 작은 노이즈 폴리곤 제거 (예: 면적이 10 미만)
                    if cv2.contourArea(contour) < 10:
                        continue
                        
                    # contour shape is (N, 1, 2) -> (N, 2)
                    points = contour.squeeze().tolist()
                    # 점이 3개 이상이어야 폴리곤 성립
                    if isinstance(points[0], list) and len(points) >= 3:
                        polygons.append({
                            "label": label_name,
                            "points": points,
                            "group_id": None,
                            "shape_type": "polygon",
                            "flags": {}
                        })
                
                # ---------------------------------------------------------
                # JSON 및 시각화 이미지 저장
                # ---------------------------------------------------------
                # AnyLabeling / LabelMe 호환 포맷
                json_data = {
                    "version": "4.0.0",
                    "flags": {},
                    "shapes": polygons,
                    "imagePath": file,
                    "imageData": None,
                    "imageHeight": h,
                    "imageWidth": w
                }
                
                with open(out_path_json, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)
                
                # 시각화 (외곽선 그리기)
                annotated_image = image.copy()
                for shape in polygons:
                    pts = np.array(shape["points"], np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    # 폴리곤 외곽선을 초록색으로 표시
                    cv2.polylines(annotated_image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                
                cv2.imwrite(out_path_img, annotated_image)
                print(f"  -> 폴리곤 {len(polygons)}개 추출 및 저장 완료: {os.path.basename(out_path_json)}")

    print(f"\n모든 작업이 완료되었습니다. 결과는 {OUTPUT_DIR} 에서 확인하세요.")

if __name__ == "__main__":
    main()
