import cv2
import numpy as np
import os
import glob

# --- 사용자 설정 ---

# 1. remapped 그레이스케일 이미지가 있는 폴더
REMAPPED_FOLDER = 'trainannot_remapped'

# 2. 원본 컬러 이미지가 있는 폴더
ORIGINAL_FOLDER = 'train'

# 3. 찾고자 하는 remapped 이미지의 픽셀 값
TARGET_VALUE = 2

# 4. 검색할 이미지 파일 확장자
IMAGE_EXTENSIONS = ('*.png', '*.jpg', '*.jpeg', '*.bmp')

# 5. 결과를 저장할 텍스트 파일 이름 (새로 추가)
OUTPUT_TXT_FILE = '3_found_bgr_colors.txt'

# --- 설정 끝 ---

def analyze_pixel_colors():
    """
    Remapped 폴더의 특정 픽셀 값 위치에 해당하는
    원본 폴더의 픽셀 색상을 분석합니다.
    """
    print(f"분석 시작: '{REMAPPED_FOLDER}' 폴더의 픽셀 값 {TARGET_VALUE}를 기준으로 '{ORIGINAL_FOLDER}' 폴더를 분석합니다.")
    
    # 발견된 고유한 BGR 색상을 저장할 Set
    found_colors = set()

    # 모든 이미지 확장자에 대해 파일 검색
    remapped_files = []
    for ext in IMAGE_EXTENSIONS:
        search_path = os.path.join(REMAPPED_FOLDER, ext)
        remapped_files.extend(glob.glob(search_path))

    if not remapped_files:
        print(f"오류: '{REMAPPED_FOLDER}'에서 이미지를 찾을 수 없습니다. 경로를 확인하세요.")
        return

    processed_count = 0
    for remapped_file_path in remapped_files:
        file_name = os.path.basename(remapped_file_path)

        # 파일 이름 규칙 변환 (e.g., annotation_00001.png -> image_00001.png)
        if not file_name.startswith('annotation_'):
            print(f"  [경고] 예상치 못한 파일 이름: {file_name} ('annotation_'로 시작하지 않음. 건너뜀)")
            continue
            
        original_file_name = file_name.replace('annotation_', 'image_', 1)
        original_file_path = os.path.join(ORIGINAL_FOLDER, original_file_name)

        # 1. 원본 파일 존재 여부 확인
        if not os.path.exists(original_file_path):
            print(f"  [경고] 원본 파일 없음: {original_file_path} (건너뜀)")
            continue

        # 2. 이미지 읽기
        remapped_img = cv2.imread(remapped_file_path, cv2.IMREAD_GRAYSCALE)
        original_img = cv2.imread(original_file_path, cv2.IMREAD_COLOR)

        # 3. 이미지 유효성 검사
        if remapped_img is None or original_img is None:
            print(f"  [경고] 이미지 로드 실패: {file_name} (건너뜀)")
            continue
            
        if remapped_img.shape[:2] != original_img.shape[:2]:
            print(f"  [경고] 이미지 크기 불일치: {file_name} (건너뜀)")
            continue

        # 4. 픽셀 값 2인 위치 찾기 (NumPy 사용)
        locations = np.where(remapped_img == TARGET_VALUE)
        
        if locations[0].size > 0:
            y_coords, x_coords = locations
            
            # 5. 원본 이미지에서 해당 위치의 BGR 값들 가져오기
            colors_at_locations = original_img[y_coords, x_coords]
            
            # 6. 고유한 색상만 추출하여 set에 추가
            unique_colors_in_this_image = np.unique(colors_at_locations, axis=0)
            
            for bgr_color in unique_colors_in_this_image:
                found_colors.add(tuple(bgr_color))
        
        processed_count += 1
    
    # 7. 최종 결과 보고 (이 부분이 수정되었습니다)
    print("\n--- 분석 완료 ---")
    print(f"총 {processed_count}개의 파일을 분석했습니다.")
    
    if not found_colors:
        print(f"결과: 값이 {TARGET_VALUE}인 픽셀에 해당하는 원본 이미지 픽셀을 찾지 못했습니다.")
    else:
        # 화면에 출력하는 대신 파일에 저장
        sorted_colors = sorted(list(found_colors))
        
        try:
            # 텍스트 파일을 쓰기('w') 모드로 엽니다.
            with open(OUTPUT_TXT_FILE, 'w', encoding='utf-8') as f:
                # 헤더(제목줄)를 추가하고 싶다면 다음 줄의 주석을 해제하세요.
                # f.write("B, G, R\n")
                
                for color in sorted_colors:
                    # color는 (B, G, R) 튜플입니다.
                    # B, G, R 형식으로 파일에 씁니다. (예: 100, 50, 20)
                    f.write(f"{color[0]}, {color[1]}, {color[2]}\n")
            
            print(f"결과: 총 {len(sorted_colors)}개의 고유한 색상을 '{OUTPUT_TXT_FILE}' 파일에 저장했습니다.")
        
        except IOError as e:
            print(f"오류: 결과를 파일({OUTPUT_TXT_FILE})에 저장하는 중 오류가 발생했습니다: {e}")
            
# --- 메인 실행 ---
if __name__ == "__main__":
    if not os.path.isdir(REMAPPED_FOLDER):
        print(f"오류: '{REMAPPED_FOLDER}' 폴더를 찾을 수 없습니다. 스크립트 상단의 경로를 확인하세요.")
    elif not os.path.isdir(ORIGINAL_FOLDER):
        print(f"오류: '{ORIGINAL_FOLDER}' 폴더를 찾을 수 없습니다. 스크립트 상단의 경로를 확인하세요.")
    else:
        analyze_pixel_colors()