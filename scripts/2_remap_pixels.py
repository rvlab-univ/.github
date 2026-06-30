import cv2
import numpy as np
import glob
import os

# --- 사용자 설정 ---

# 1. 원본 이미지들이 있는 폴더 경로
SOURCE_FOLDER = 'C:/Users/yang/Desktop/SeRM Dataset/0.SeRM_image-002/trainannot'

# 2. 변환된 이미지들을 저장할 폴더 경로 (원본과 다른 곳을 추천)
#    폴더가 없으면 자동으로 생성됩니다.
OUTPUT_FOLDER = 'C:/Users/yang/Desktop/SeRM Dataset/0.SeRM_image-002/trainannot_remapped'

# --- 설정 끝 ---


# 3. 픽셀 값 매핑 딕셔너리 (사용자 요청 기반)
#    old_value : new_value
pixel_map = {
    1: 13,
    2: 6,
    3: 8,
    4: 7,
    5: 10,
    6: 9,
    7: 5,
    8: 3,
    9: 1,
    10: 2,
    11: 3,
    12: 2,
    13: 4,
    14: 11,
    15: 12,
    16: 14
}

def create_lookup_table(mapping):
    """
    픽셀 값 매핑 딕셔너리를 기반으로 OpenCV용 Look-Up Table (LUT)을 생성합니다.
    """
    # 1. 0부터 255까지의 값을 갖는 배열을 생성 (기본값: 0->0, 1->1, 2->2, ...)
    #    8비트 그레이스케일 이미지(0-255)이므로 256 크기
    lut = np.arange(256, dtype=np.uint8)
    
    # 2. 딕셔너리에 정의된 값들로 LUT를 업데이트합니다.
    for old_val, new_val in mapping.items():
        if 0 <= old_val < 256:
             lut[old_val] = new_val
             
    return lut

def process_images_in_folder(source_dir, output_dir, lut):
    """
    지정된 폴더의 모든 PNG 이미지를 LUT를 사용해 변환하고 새 폴더에 저장합니다.
    """
    # 출력 폴더가 없으면 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 소스 폴더 내의 모든 .png 파일을 찾습니다.
    # os.path.join을 사용하여 OS 호환 경로를 만듭니다.
    search_path = os.path.join(source_dir, '*.png')
    image_files = glob.glob(search_path)
    
    if not image_files:
        print(f"경고: '{source_dir}'에서 PNG 파일을 찾을 수 없습니다.")
        return

    print(f"총 {len(image_files)}개의 PNG 파일을 처리합니다...")

    for file_path in image_files:
        try:
            # 1. 이미지를 그레이스케일로 읽어옵니다.
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                print(f"  [실패] '{file_path}' 파일을 읽을 수 없습니다. 건너뜁니다.")
                continue

            # 2. LUT를 이미지에 적용합니다. (가장 핵심적인 부분)
            #    img의 모든 픽셀 값을 lut의 인덱스로 사용하여 새 값으로 치환합니다.
            img_remapped = cv2.LUT(img, lut)
            
            # 3. 결과 이미지를 저장합니다.
            file_name = os.path.basename(file_path)
            output_path = os.path.join(output_dir, file_name)
            
            cv2.imwrite(output_path, img_remapped)
            print(f"  [성공] {file_name} -> {output_path} (저장 완료)")

        except Exception as e:
            print(f"  [오류] {file_path} 처리 중 오류 발생: {e}")

    print("\n모든 작업이 완료되었습니다.")

# --- 메인 실행 ---
if __name__ == "__main__":
    # 1. 변환 테이블(LUT) 생성
    lookup_table = create_lookup_table(pixel_map)
    
    # 2. 이미지 처리 함수 실행
    process_images_in_folder(SOURCE_FOLDER, OUTPUT_FOLDER, lookup_table)