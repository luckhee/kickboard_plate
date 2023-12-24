import torch
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.datasets import LoadImages
from utils.plots import plot_one_box
import cv2

# 사용할 장치 설정 (CUDA가 가능하면 CUDA 사용)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 모델 로드 (모델 파일 경로를 로컬 시스템에 맞게 수정해야 합니다)
model = attempt_load('/Volumes/Hee/kicboard/yolov7/kicboard.pt', map_location=device)  # 예시: 'C:/models/last.pt'
model.eval()

# 클래스 이름 로드 (이 부분을 자신의 데이터셋에 맞게 수정해야 합니다)
names = model.module.names if hasattr(model, 'module') else model.names

# 이미지나 비디오 파일 로드 (파일 경로를 로컬 시스템에 맞게 수정해야 합니다)
path = '/Volumes/Hee/kicboard/yolov7/google-images-0_jpg.rf.2a21d41ab9cbcfd2e393328e555266dd.jpg'  # 예시: 'C:/images/myimage.jpeg'
dataset = LoadImages(path, img_size=640)

# 추론 및 결과 시각화
for path, img, im0s, _ in dataset:
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 이미지를 float32로 변환
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # 추론
    with torch.no_grad():
        pred = model(img)[0]

    # NMS 적용
    pred = non_max_suppression(pred, 0.4, 0.5, classes=None, agnostic=False)

    # 탐지된 객체 시각화
    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, im0s, label=label, color=(255, 0, 0), line_thickness=3)

    # 결과 이미지를 화면에 표시
    cv2.imshow('Result', im0s)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
