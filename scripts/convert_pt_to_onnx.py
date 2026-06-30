import sys
import argparse
from ultralytics import YOLO

def export_yolo_to_onnx(pt_path):
    """
    YOLOv8 / YOLO11 (.pt) 모델을 ONNX 형식으로 변환합니다.
    """
    print(f"[{pt_path}] 로딩 중...")
    try:
        model = YOLO(pt_path)
    except Exception as e:
        print(f"모델 로딩 실패: {e}")
        return

    print("ONNX 형식으로 변환을 시작합니다...")
    # opset이나 imgsz 등 추가 옵션이 필요하다면 아래 export 함수에 추가할 수 있습니다.
    success = model.export(format='onnx')
    
    if success:
        print(f"변환 성공! 결과물이 저장되었습니다.")
    else:
        print("변환에 실패했습니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch (.pt) 파일을 ONNX로 변환하는 스크립트 (YOLO 모델 전용)")
    parser.add_argument("pt_path", type=str, help="변환할 .pt 파일의 경로 (예: best.pt)")
    
    args = parser.parse_args()
    export_yolo_to_onnx(args.pt_path)
