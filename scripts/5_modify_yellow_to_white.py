import cv2
import numpy as np
import os
import glob

# --- 사용자 설정 ---

# 1. remapped 그레이스케일 이미지가 있는 폴더
REMAPPED_FOLDER = 'trainannot_remapped'

# 2. 원본 컬러 이미지가 있는 폴더
ORIGINAL_FOLDER = 'train'

# 3. 결과 이미지를 저장할 새 폴더 (자동으로 생성됨)
OUTPUT_FOLDER = 'train_desaturated_v3_skipped'

# 4. 색상을 변경할 기준 픽셀 값
TARGET_VALUE = 2

# 5. '유채색'으로 판단할 채도(S) 임계값 (이 값 *초과*시 변경 대상)
GRAYSCALE_S_THRESHOLD = 20

# 6. [v2] 채도를 얼마나 줄일지 (비례 값)
REDUCTION_FACTOR = 0.1

# 7. [v2] 축소 후 채도의 최대값 (이 값 이하로 제한)
MAX_S_AFTER_REDUCTION = 20

# --- 설정 끝 ---

def desaturate_target_pixels_v3():
    """
    Remapped 폴더의 특정 픽셀 값 위치에 해당하는
    원본 폴더의 '유채색' 픽셀만 '비례적으로 채도를 낮춥니다'.
    TARGET_VALUE 픽셀이 없는 이미지는 스킵합니다.
    """
    print(f"작업 시작: '{REMAPPED_FOLDER}' 폴더의 값 {TARGET_VALUE} 픽셀을 기준으로 합니다.")
    print(f"설정: S > {GRAYSCALE_S_THRESHOLD}인 픽셀의 채도를 {REDUCTION_FACTOR*100}%로 줄임 (최대 {MAX_S_AFTER_REDUCTION})")
    print(f"'{OUTPUT_FOLDER}' 폴더에 저장됩니다. (TARGET_VALUE가 없는 이미지는 스킵)")
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    search_path = os.path.join(REMAPPED_FOLDER, '*.png')
    remapped_files = glob.glob(search_path)

    if not remapped_files:
        print(f"오류: '{REMAPPED_FOLDER}'에서 이미지를 찾을 수 없습니다.")
        return

    processed_count = 0 # 총 처리(스캔)한 파일 수
    saved_count = 0     # 스킵하지 않고 *저장*한 파일 수
    
    for remapped_file_path in remapped_files:
        processed_count += 1
        file_name = os.path.basename(remapped_file_path)
        
        if not file_name.startswith('annotation_'):
            print(f"  [경고] {file_name}: 'annotation_'로 시작하지 않음 (건너뜀)")
            continue
            
        original_file_name = file_name.replace('annotation_', 'image_', 1)
        original_file_path = os.path.join(ORIGINAL_FOLDER, original_file_name)

        if not os.path.exists(original_file_path):
            print(f"  [경고] 원본 파일 없음: {original_file_path} (건너뜀)")
            continue

        # 2. 이미지 읽기
        remapped_img = cv2.imread(remapped_file_path, cv2.IMREAD_GRAYSCALE)
        original_img = cv2.imread(original_file_path, cv2.IMREAD_COLOR)

        if remapped_img is None or original_img is None:
            print(f"  [경고] 이미지 로드 실패: {file_name} (건너뜀)")
            continue
            
        if remapped_img.shape[:2] != original_img.shape[:2]:
            print(f"  [경고] 이미지 크기 불일치: {file_name} (건너뜀)")
            continue

        # --- 핵심 로직 시작 (v3) ---

        # 3. remapped 값이 TARGET_VALUE인 위치 탐색
        mask_target_value = (remapped_img == TARGET_VALUE)
        
        # 4. [수정] TARGET_VALUE 픽셀이 하나라도 있는지 확인
        if not np.any(mask_target_value):
            # 하나도 없으면 이 이미지는 스킵 (저장 안 함)
            if processed_count <= 20: # 스킵 로그는 처음 20개만 출력
                print(f"  [스킵] {file_name}: TARGET_VALUE({TARGET_VALUE}) 픽셀 없음.")
            elif processed_count == 21:
                print("  ... (이후 스킵 로그는 생략) ...")
            continue # 다음 파일로
        
        # --- (여기부터는 TARGET_VALUE 픽셀이 *있는* 이미지들만 처리) ---
        
        # 5. HSV로 변환
        hsv_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
        
        # 6. '유채색'인 위치 마스크 (S > Threshold)
        s_channel = hsv_img[:, :, 1] 
        mask_is_colorful = (s_channel > GRAYSCALE_S_THRESHOLD)
        
        # 7. 두 조건을 *모두* 만족하는 최종 마스크
        #    (TARGET_VALUE 이면서 동시에 유채색인 곳)
        final_mask = mask_target_value & mask_is_colorful

        # 8. 변경 적용
        if np.any(final_mask): # (값이 2이고 *유채색*인 픽셀이 있다면)
            
            s_channel_float = s_channel.astype(np.float32)
            s_reduced = s_channel_float * REDUCTION_FACTOR 
            s_clipped = np.clip(s_reduced, 0, MAX_S_AFTER_REDUCTION) 
            s_channel_new = np.where(final_mask, s_clipped, s_channel_float)
            hsv_img[:, :, 1] = s_channel_new.astype(np.uint8)
            
            # HSV를 BGR로 다시 변환
            output_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
        
        else: # (값이 2인 픽셀은 *있지만*, 모두 *무채색*이라면)
            output_img = original_img # 원본 그대로

        # 9. 결과 저장 (if 블록 밖에 있으므로 *항상* 저장됨)
        output_path = os.path.join(OUTPUT_FOLDER, original_file_name)
        cv2.imwrite(output_path, output_img)
        
        saved_count += 1
        if saved_count % 50 == 0:
            print(f"  ... {saved_count}개 저장 완료 (총 {processed_count}개 스캔) ...")
        
        # --- 핵심 로직 끝 ---

    print("\n--- 모든 작업 완료 ---")
    print(f"총 {processed_count}개의 이미지를 스캔했습니다.")
    print(f"총 {saved_count}개의 이미지를 '{OUTPUT_FOLDER}'에 저장했습니다. ({processed_count - saved_count}개 스킵됨)")

# --- 메인 실행 ---
if __name__ == "__main__":
    if not os.path.isdir(REMAPPED_FOLDER):
        print(f"오류: '{REMAPPED_FOLDER}' 폴더를 찾을 수 없습니다.")
    elif not os.path.isdir(ORIGINAL_FOLDER):
        print(f"오류: '{ORIGINAL_FOLDER}' 폴더를 찾을 수 없습니다.")
    else:
        desaturate_target_pixels_v3()