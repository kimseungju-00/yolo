import cv2
import numpy as np
from ultralytics import YOLO

# YOLO 모델 로드
model = YOLO('yolov8n.pt')  # YOLOv8의 n 모델 사용, 필요에 따라 다른 모델 사용 가능

# 동영상 파일 열기
cap = cv2.VideoCapture('v3.mp4')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
output = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

# Gaussian 디블러링 필터
def apply_deblurring(image, bbox):
    (x, y, w, h) = bbox
    object_area = image[y:y+h, x:x+w]  # 객체 영역 추출
    deblurred_object = cv2.GaussianBlur(object_area, (5, 5), 0)  # Gaussian 블러링 적용
    image[y:y+h, x:x+w] = deblurred_object  # 원본 이미지에 반영
    return image

# 프레임 단위로 처리
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 객체 탐지 수행
    results = model(frame)

    # 탐지된 객체 반복
    for obj in results:
        for det in obj.boxes.data:
            x1, y1, x2, y2, conf, cls = map(int, det.tolist())  # bbox 좌표 및 클래스 정보 추출

            # 특정 클래스만 디블러링 처리 (예: 0은 person 클래스)
            if cls == 0:  
                bbox = (x1, y1, x2 - x1, y2 - y1)
                frame = apply_deblurring(frame, bbox)  # 디블러링 필터 적용

    # 결과 프레임을 저장
    output.write(frame)
    cv2.imshow('Deblurred Frame', frame)
    
    # 'q'를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
output.release()
cv2.destroyAllWindows()
