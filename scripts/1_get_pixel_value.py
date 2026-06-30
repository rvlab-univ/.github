import cv2
import sys
import os

# --- 설정 ---
# 여기에 픽셀 값을 확인할 이미지 파일 경로를 입력하세요.
# (예: 'C:/Users/User/Desktop/my_image.png')
# 백슬래시(\) 대신 슬래시(/)를 사용하거나 이중 백슬래시(\\)를 사용하세요.
IMAGE_PATH = 'trainannot_remapped/annotation_00137.png' 
# --- 설정 끝 ---

# 이미지를 그레이스케일로 불러옵니다.
# cv2.IMREAD_GRAYSCALE 플래그는 이미지를 1채널 그레이스케일로 강제 변환합니다.
img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

# 이미지 로드 실패 시 예외 처리
if img is None:
    print(f"Error: 이미지를 로드할 수 없습니다.")
    print(f"경로를 확인하세요: {os.path.abspath(IMAGE_PATH)}")
    sys.exit()

# 마우스 클릭 이벤트를 처리할 콜백 함수
def mouse_callback(event, x, y, flags, param):
    """
    마우스 이벤트가 발생할 때 호출되는 함수
    event: 발생한 이벤트 (예: cv2.EVENT_LBUTTONDOWN)
    x, y: 이벤트가 발생한 픽셀 좌표
    flags, param: OpenCV에서 사용하는 추가 파라미터
    """
    
    # 왼쪽 마우스 버튼을 클릭했을 때
    if event == cv2.EVENT_LBUTTONDOWN:
        # img[y, x] 순서로 픽셀 값에 접근합니다. (세로, 가로)
        pixel_value = img[y, x]
        
        # Anaconda Prompt 창에 좌표와 픽셀 값을 출력합니다.
        print(f"클릭 좌표 (x={x}, y={y}) | 픽셀 값: {pixel_value}")

# 메인 코드
window_name = 'Grayscale Image - Click to get pixel value'
cv2.namedWindow(window_name)

# 'window_name'이라는 이름의 창에 mouse_callback 함수를 연결합니다.
cv2.setMouseCallback(window_name, mouse_callback)

print("--- 이미지 픽셀 값 확인기 ---")
print(f"이미지 로드 완료: {IMAGE_PATH} (크기: {img.shape[1]}x{img.shape[0]})")
print("이미지 창에서 원하는 위치를 클릭하세요.")
print("종료하려면 'q' 또는 'Esc' 키를 누르세요.")

while True:
    # 이미지를 화면에 표시합니다.
    cv2.imshow(window_name, img)
    
    # 1ms 동안 키 입력을 기다립니다.
    key = cv2.waitKey(1) & 0xFF
    
    # 'q' 키 또는 'Esc' (ASCII 27) 키를 누르면 루프를 종료합니다.
    if key == ord('q') or key == 27:
        print("프로그램을 종료합니다.")
        break

# 모든 OpenCV 창을 닫습니다.
cv2.destroyAllWindows()