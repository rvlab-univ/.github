import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# --- 사용자 설정 ---

# 1. BGR 값이 저장된 입력 텍스트 파일
INPUT_FILE = '4_classified_yellow_colors.txt'

# 2. 생성될 히스토그램 이미지 파일
OUTPUT_HISTOGRAM_IMAGE = '4_saturation_histogram.png'

# --- 설정 끝 ---

def analyze_saturation_distribution():
    """
    BGR 텍스트 파일을 읽어 S(채도) 값의 분포를 분석하고 
    히스토그램을 저장합니다.
    """
    if not os.path.exists(INPUT_FILE):
        print(f"오류: 입력 파일 '{INPUT_FILE}'을(를) 찾을 수 없습니다.")
        return

    print(f"파일을 읽는 중입니다: {INPUT_FILE}")
    
    bgr_colors_list = []
    
    # 1. 파일 읽기 및 파싱
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(',')
                if len(parts) == 3:
                    try:
                        b = int(parts[0].strip())
                        g = int(parts[1].strip())
                        r = int(parts[2].strip())
                        # (B, G, R) 순서로 리스트에 추가
                        bgr_colors_list.append([b, g, r]) 
                    except ValueError:
                        pass # 헤더나 잘못된 줄은 무시
                        
    except Exception as e:
        print(f"파일 처리 중 오류 발생: {e}")
        return

    if not bgr_colors_list:
        print("오류: 파일에서 유효한 BGR 값을 읽지 못했습니다.")
        return

    print(f"총 {len(bgr_colors_list)}개의 BGR 값을 읽었습니다. HSV로 변환 중...")

    # 2. BGR -> HSV 일괄 변환 (매우 빠름)
    # 리스트를 (1, N, 3) 형태의 NumPy 배열로 변환
    bgr_array = np.uint8([bgr_colors_list])
    
    # BGR -> HSV 변환
    hsv_array = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2HSV)
    
    # S(채도) 값만 추출 (0~255 범위)
    # hsv_array[0]은 (N, 3) 배열, [:, 1]은 모든 N개의 S값을 의미
    s_values = hsv_array[0, :, 1]
    
    print("변환 완료. 통계 분석 및 히스토그램 생성 중...")

    # 3. 통계 분석
    total_count = len(s_values)
    
    print("\n--- 채도(Saturation) 값 분포 통계 ---")
    
    # S=0 (완전 무채색)인 픽셀 수
    s_zero_count = np.sum(s_values == 0)
    print(f"  S = 0 (완전 무채색):   {s_zero_count:7d} 개 ({s_zero_count/total_count:6.2%})")

    # S가 특정 값 이하인 픽셀 수 (누적)
    for threshold in [10, 20, 30, 40, 50]:
        s_low_count = np.sum(s_values <= threshold)
        print(f"  S <= {threshold} (거의 무채색): {s_low_count:7d} 개 ({s_low_count/total_count:6.2%})")
        
    print("------------------------------------------")

    # 4. 히스토그램 생성 및 저장
    try:
        plt.figure(figsize=(12, 7))
        
        # 0~255 범위를 50개 구간으로 나누어 히스토그램 생성
        # bins=256으로 하면 너무 촘촘하므로 50~100개가 적당
        plt.hist(s_values, bins=50, range=(0, 255), color='blue', alpha=0.7, edgecolor='black')
        
        plt.title('Saturation (S) Value Distribution (채도 값 분포)', fontsize=16)
        plt.xlabel('Saturation Value (S) (0=Grayscale, 255=Full Color)', fontsize=12)
        plt.ylabel('Frequency (픽셀 수)', fontsize=12)
        
        # S=0 근처를 잘 보기 위해 Y축을 log scale로 변경 (선택 사항)
        # S=0 근처에 픽셀이 너무 많으면 다른 구간이 안보임
        plt.yscale('log')
        plt.grid(True, axis='y', linestyle='--', alpha=0.6)
        
        # S 값 20, 30, 40에 기준선 표시
        plt.axvline(x=20, color='red', linestyle='--', label='S = 20')
        plt.axvline(x=30, color='green', linestyle='--', label='S = 30')
        plt.axvline(x=40, color='orange', linestyle='--', label='S = 40')
        plt.legend()
        
        plt.savefig(OUTPUT_HISTOGRAM_IMAGE)
        plt.close() # 메모리 해제
        
        print(f"\n히스토그램이 '{OUTPUT_HISTOGRAM_IMAGE}' 파일로 저장되었습니다.")
        print("그래프에서 S=0 근처의 '무채색' 그룹과 분리되는 '계곡' 지점을 찾아보세요.")
        print("그 계곡 지점이 '유채색'을 판단하는 좋은 Threshold가 될 수 있습니다.")

    except Exception as e:
        print(f"히스토그램 생성 중 오류 발생: {e}")
        print("matplotlib 라이브러리가 설치되어 있는지 확인하세요. (pip install matplotlib)")

# --- 메인 실행 ---
if __name__ == "__main__":
    # Anaconda Prompt에서 opencv-python, numpy, matplotlib가 설치되어 있어야 합니다.
    # pip install opencv-python numpy matplotlib
    analyze_saturation_distribution()