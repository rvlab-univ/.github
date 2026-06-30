import os
import cv2
import json
import torch
import numpy as np
import shutil
from tqdm import tqdm
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def main():
    # -----------------------------------------------------
    # 1. 설정 및 경로
    # -----------------------------------------------------
    SAM2_CHECKPOINT = "/home/lim/test_ws/checkpoint.pt"
    SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 입력/출력 디렉토리 설정
    INPUT_DIR = "/home/lim/test_ws/datasets/kaist_data/images_2000"
    OUTPUT_DIR = "/home/lim/test_ws/datasets/kaist_data/images_2000_polygons"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # -----------------------------------------------------
    # 2. SAM2 모델 초기화
    # -----------------------------------------------------
    print(f"SAM2 모델 로드 중 (디바이스: {DEVICE})...")
    sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
    predictor = SAM2ImagePredictor(sam2_model)
    
    # -----------------------------------------------------
    # 3. 전체 이미지 파일 목록 수집
    # -----------------------------------------------------
    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"총 {len(image_files)}장의 이미지에 대해 폴리곤 자동 라벨링을 시작합니다.")
    print(f"결과물 저장 폴더: {OUTPUT_DIR}\n")
    
    # -----------------------------------------------------
    # 4. 순회 및 일괄 추론
    # -----------------------------------------------------
    for img_filename in tqdm(image_files, desc="Processing Images"):
        img_path = os.path.join(INPUT_DIR, img_filename)
        json_filename = os.path.splitext(img_filename)[0] + ".json"
        json_path = os.path.join(INPUT_DIR, json_filename)
        
        out_json_path = os.path.join(OUTPUT_DIR, json_filename)
        out_img_path = os.path.join(OUTPUT_DIR, img_filename)
        
        # 원본 이미지 파일 복사 (AnyLabeling에서 바로 열어볼 수 있도록)
        if not os.path.exists(out_img_path):
            shutil.copy(img_path, out_img_path)
        
        if not os.path.exists(json_path):
            continue
            
        # 원본 이미지 로드
        image = cv2.imread(img_path)
        if image is None:
            continue
            
        # JSON에서 BBox 정보 추출
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        bboxes = []
        labels = []
        
        for shape in data.get('shapes', []):
            if shape.get('shape_type') != 'rectangle':
                continue
            
            pts = shape['points']
            x_coords = [p[0] for p in pts]
            y_coords = [p[1] for p in pts]
            
            # BBox: [x_min, y_min, x_max, y_max]
            bboxes.append([min(x_coords), min(y_coords), max(x_coords), max(y_coords)])
            labels.append(shape['label'])
            
        # BBox가 없는 이미지라면 원본 JSON을 그대로 복사 후 넘어감
        if not bboxes:
            with open(out_json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            continue
            
        # -----------------------------------------------------
        # 핵심 로직: 원본 이미지에 전체 BBox들을 한 번에 프롬프트
        # -----------------------------------------------------
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)
        
        input_boxes = np.array(bboxes)
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False
        )
        
        new_shapes = []
        
        # 반환된 마스크들을 폴리곤으로 변환
        for i, mask in enumerate(masks):
            mask_2d = mask[0] if mask.ndim == 3 else mask
            mask_uint8 = (mask_2d * 255).astype(np.uint8)
            
            # 외곽선 추출
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # 너무 작은 노이즈 제거 (면적 20 미만)
                if cv2.contourArea(contour) < 20: 
                    continue
                
                # 선택사항: 좌표 수를 줄이기 위한 다각형 근사 (필요 시 주석 해제)
                # epsilon = 0.002 * cv2.arcLength(contour, True)
                # contour = cv2.approxPolyDP(contour, epsilon, True)
                
                points = contour.squeeze().tolist()
                
                # 점이 3개 이상 모여야 유효한 폴리곤
                if isinstance(points[0], list) and len(points) >= 3:
                    new_shapes.append({
                        "label": labels[i],
                        "points": points,
                        "group_id": None,
                        "shape_type": "polygon",
                        "flags": {}
                    })
                    
        # -----------------------------------------------------
        # 5. 폴리곤으로 대체된 새로운 JSON 저장
        # -----------------------------------------------------
        data['shapes'] = new_shapes
        with open(out_json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
