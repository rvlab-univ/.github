import os
import json
import shutil
import random
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import pycocotools.mask as mask_util

def polygon_to_rle(polygon, height, width):
    """
    Polygon 좌표를 RLE 형식으로 변환합니다.
    
    Args:
        polygon: [[x1, y1, x2, y2, ...]] 형식의 폴리곤 좌표 리스트
        height: 이미지 높이
        width: 이미지 너비
    
    Returns:
        RLE 딕셔너리 (counts, size)
    """
    # polygon을 numpy 배열로 변환
    if isinstance(polygon, list):
        if len(polygon) == 0:
            return None
        # polygon이 중첩 리스트인 경우 평탄화
        if isinstance(polygon[0], list):
            polygon = [coord for sublist in polygon for coord in sublist]
        polygon = np.array(polygon, dtype=np.float32)
    
    # polygon을 정수 좌표로 변환
    polygon = polygon.reshape(-1, 2).astype(np.int32)
    
    # 마스크 생성
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 1)
    
    # RLE 인코딩
    rle = mask_util.encode(np.asfortranarray(mask))
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle

def scale_json_annotations(input_json_path, output_json_path, original_width, original_height, new_width, new_height):
    """
    COCO 형식 JSON 어노테이션 파일을 스케일링합니다.
    Polygon segmentation을 RLE로 변환하고, bbox와 area를 스케일링합니다.
    
    Args:
        input_json_path: 원본 JSON 파일 경로
        output_json_path: 출력 JSON 파일 경로
        original_width: 원본 이미지 너비
        original_height: 원본 이미지 높이
        new_width: 새 이미지 너비
        new_height: 새 이미지 높이
    """
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    except Exception as e:
        print(f"오류: JSON 파일 '{input_json_path}' 로드 실패: {e}")
        return False
    
    # 스케일 팩터 계산
    width_scale = new_width / original_width
    height_scale = new_height / original_height
    
    # 이미지 정보 업데이트
    if 'images' in json_data and len(json_data['images']) > 0:
        json_data['images'][0]['height'] = new_height
        json_data['images'][0]['width'] = new_width
        # 파일명 확장자 변경 (.png -> .jpg로 변경할 수도 있지만, 일단 유지)
    
    # 어노테이션 스케일링
    updated_annotations = []
    for ann in json_data.get('annotations', []):
        # 바운딩 박스 스케일링
        if 'bbox' in ann and len(ann['bbox']) == 4:
            x, y, w, h = ann['bbox']
            ann['bbox'] = [
                x * width_scale,
                y * height_scale,
                w * width_scale,
                h * height_scale
            ]
        
        # 면적 스케일링
        if 'area' in ann:
            ann['area'] = ann['area'] * width_scale * height_scale
        
        # Segmentation 처리
        if 'segmentation' in ann:
            seg = ann['segmentation']
            
            # Polygon 형식인 경우 RLE로 변환
            if isinstance(seg, list) and len(seg) > 0:
                # Polygon 좌표를 스케일링
                scaled_polygon = []
                for poly in seg:
                    if isinstance(poly, list):
                        scaled_poly = []
                        for i in range(0, len(poly), 2):
                            if i + 1 < len(poly):
                                x = poly[i] * width_scale
                                y = poly[i + 1] * height_scale
                                scaled_poly.extend([x, y])
                        scaled_polygon.append(scaled_poly)
                
                # 스케일링된 polygon을 RLE로 변환
                if len(scaled_polygon) > 0:
                    # 첫 번째 polygon 사용 (여러 polygon이 있으면 합침)
                    combined_polygon = []
                    for poly in scaled_polygon:
                        combined_polygon.extend(poly)
                    
                    rle = polygon_to_rle(combined_polygon, new_height, new_width)
                    if rle is not None:
                        ann['segmentation'] = rle
                    else:
                        print(f"  경고: Annotation ID {ann.get('id', 'N/A')}의 RLE 변환 실패")
                        continue
                else:
                    print(f"  경고: Annotation ID {ann.get('id', 'N/A')}의 polygon이 비어있음")
                    continue
            # 이미 RLE 형식인 경우
            elif isinstance(seg, dict) and 'counts' in seg and 'size' in seg:
                try:
                    rle = {
                        'counts': seg['counts'],
                        'size': [original_height, original_width]
                    }
                    mask = mask_util.decode(rle)
                    resized_mask = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
                    new_rle = mask_util.encode(np.asfortranarray(resized_mask))
                    new_rle['counts'] = new_rle['counts'].decode('utf-8')
                    ann['segmentation'] = new_rle
                except Exception as e:
                    print(f"  경고: Annotation ID {ann.get('id', 'N/A')}의 RLE 처리 실패: {e}")
                    continue
        
        updated_annotations.append(ann)
    
    json_data['annotations'] = updated_annotations
    
    # JSON 파일 저장
    try:
        Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"오류: JSON 파일 저장 실패 '{output_json_path}': {e}")
        return False

def resize_image(input_image_path, output_image_path, output_size=(1024, 1024)):
    """
    이미지를 리사이즈합니다.
    
    Args:
        input_image_path: 원본 이미지 경로
        output_image_path: 출력 이미지 경로
        output_size: (width, height) 튜플
    """
    try:
        with Image.open(input_image_path) as img:
            if img.size == output_size:
                # 이미 목표 크기면 복사만
                shutil.copy2(input_image_path, output_image_path)
                return True
            
            resized_img = img.resize(output_size, Image.LANCZOS)
            resized_img.save(output_image_path, quality=95)
            return True
    except Exception as e:
        print(f"오류: 이미지 리사이즈 실패 '{input_image_path}': {e}")
        return False

def process_dataset(
    input_base_dir,
    output_base_dir,
    original_width=1280,
    original_height=320,
    target_width=1024,
    target_height=1024,
    train_ratio=0.8,
    test_ratio=0.1,
    valid_ratio=0.1,
    seed=42
):
    """
    1_serm 데이터셋을 리사이즈하고 train/test/valid로 분할합니다.
    
    Args:
        input_base_dir: 1_serm 폴더 경로 (images, labels 하위 폴더 포함)
        output_base_dir: 출력 기본 디렉토리
        original_width: 원본 이미지 너비
        original_height: 원본 이미지 높이
        target_width: 목표 이미지 너비
        target_height: 목표 이미지 높이
        train_ratio: train 비율
        test_ratio: test 비율
        valid_ratio: valid 비율
        seed: 랜덤 시드
    """
    random.seed(seed)
    np.random.seed(seed)
    
    images_dir = os.path.join(input_base_dir, "images")
    labels_dir = os.path.join(input_base_dir, "labels")
    
    if not os.path.isdir(images_dir):
        print(f"오류: 이미지 폴더를 찾을 수 없습니다: {images_dir}")
        return
    
    if not os.path.isdir(labels_dir):
        print(f"오류: 라벨 폴더를 찾을 수 없습니다: {labels_dir}")
        return
    
    # 이미지 파일 목록 수집
    image_files = []
    for filename in os.listdir(images_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            base_name = os.path.splitext(filename)[0]
            label_file = base_name + '.json'
            if os.path.exists(os.path.join(labels_dir, label_file)):
                image_files.append((filename, base_name))
    
    total_files = len(image_files)
    print(f"\n총 {total_files}개의 이미지-라벨 쌍을 찾았습니다.")
    
    if total_files == 0:
        print("처리할 파일이 없습니다.")
        return
    
    # 파일을 랜덤하게 섞기
    random.shuffle(image_files)
    
    # train/test/valid 분할
    train_end = int(total_files * train_ratio)
    test_end = train_end + int(total_files * test_ratio)
    
    train_files = image_files[:train_end]
    test_files = image_files[train_end:test_end]
    valid_files = image_files[test_end:]
    
    print(f"분할 결과:")
    print(f"  Train: {len(train_files)}개 ({len(train_files)/total_files*100:.1f}%)")
    print(f"  Test: {len(test_files)}개 ({len(test_files)/total_files*100:.1f}%)")
    print(f"  Valid: {len(valid_files)}개 ({len(valid_files)/total_files*100:.1f}%)")
    
    # 각 세트 처리
    splits = {
        'train': train_files,
        'test': test_files,
        'valid': valid_files
    }
    
    for split_name, files in splits.items():
        print(f"\n--- {split_name.upper()} 세트 처리 중 ---")
        split_output_dir = os.path.join(output_base_dir, split_name)
        os.makedirs(split_output_dir, exist_ok=True)
        
        processed = 0
        failed = 0
        
        for image_filename, base_name in files:
            input_image_path = os.path.join(images_dir, image_filename)
            input_label_path = os.path.join(labels_dir, base_name + '.json')
            
            output_image_path = os.path.join(split_output_dir, image_filename)
            output_label_path = os.path.join(split_output_dir, base_name + '.json')
            
            # 이미지 리사이즈
            if not resize_image(input_image_path, output_image_path, (target_width, target_height)):
                failed += 1
                continue
            
            # JSON 어노테이션 스케일링
            if not scale_json_annotations(
                input_label_path,
                output_label_path,
                original_width,
                original_height,
                target_width,
                target_height
            ):
                failed += 1
                continue
            
            processed += 1
            if processed % 100 == 0:
                print(f"  처리 완료: {processed}/{len(files)}")
        
        print(f"  {split_name.upper()} 완료: {processed}개 성공, {failed}개 실패")
    
    print(f"\n=== 전체 처리 완료 ===")
    print(f"출력 디렉토리: {output_base_dir}")

if __name__ == "__main__":
    # 설정
    INPUT_BASE_DIR = "/home/mscautoev/test_ws/sam2.1_finetune/data/1_serm"
    OUTPUT_BASE_DIR = "/home/mscautoev/test_ws/sam2.1_finetune/data/1_serm_processed"
    
    # 원본 이미지 해상도
    ORIGINAL_WIDTH = 1280
    ORIGINAL_HEIGHT = 320
    
    # 목표 해상도
    TARGET_WIDTH = 1024
    TARGET_HEIGHT = 1024
    
    # 분할 비율
    TRAIN_RATIO = 0.8
    TEST_RATIO = 0.1
    VALID_RATIO = 0.1
    
    # 실행
    process_dataset(
        input_base_dir=INPUT_BASE_DIR,
        output_base_dir=OUTPUT_BASE_DIR,
        original_width=ORIGINAL_WIDTH,
        original_height=ORIGINAL_HEIGHT,
        target_width=TARGET_WIDTH,
        target_height=TARGET_HEIGHT,
        train_ratio=TRAIN_RATIO,
        test_ratio=TEST_RATIO,
        valid_ratio=VALID_RATIO,
        seed=42
    )
