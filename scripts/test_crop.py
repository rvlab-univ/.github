import json
import cv2
import os

def crop_images_by_label(json_path, image_path, output_base_dir):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지를 불러올 수 없습니다: {image_path}")
        return

    base_filename = os.path.splitext(os.path.basename(image_path))[0]

    for idx, shape in enumerate(data.get('shapes', [])):
        label = shape['label']
        points = shape['points']
        
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        h, w = image.shape[:2]
        x_min, x_max = max(0, x_min), min(w, x_max)
        y_min, y_max = max(0, y_min), min(h, y_max)
        
        cropped_img = image[y_min:y_max, x_min:x_max]
        
        label_dir = os.path.join(output_base_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        
        save_name = f"{base_filename}_{label}_{idx}.png"
        save_path = os.path.join(label_dir, save_name)
        
        cv2.imwrite(save_path, cropped_img)
        print(f"저장 완료: {save_path}")

if __name__ == "__main__":
    dataset_dir = "/home/lim/test_ws/datasets/kaist_data/images_2000"
    json_file = os.path.join(dataset_dir, "image_000176.json")
    img_file = os.path.join(dataset_dir, "image_000176.png")
    
    # 워크스페이스 하위의 새로운 독립적인 폴더로 저장 경로 지정
    output_dir = "/home/lim/test_ws/cropped_test_000176"
    print(f"저장 폴더: {output_dir}")
    
    crop_images_by_label(json_file, img_file, output_dir)
