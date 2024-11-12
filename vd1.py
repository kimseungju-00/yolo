import cv2
import torch
import time
import os
os.environ['OPENCV_FFMPEG_READ_ATTEMPTS'] = '60000'
# YOLO 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)

# 동영상 파일 열기 (HEVC 코덱 사용)
cap = cv2.VideoCapture("output_h264.mp4")
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"hevc"))

# 거리 측정을 위한 함수 (가정: 공의 실제 지름 6.5cm)
def calculate_distance(width, focalLength):
    known_width = 0.065  # 공의 실제 지름 (미터)
    return (known_width * focalLength) / width

# 속도 계산 함수
def calculate_speed(distance, time):
    speed = distance / time
    return speed * 3.6  # m/s to km/h

# 공 궤적 추적 및 속도 계산
ball_trajectories = []
focalLength = None  # 초점 거리 (추후 계산)

while True:
    ret, frame = cap.read()
    if not ret:
        break  # 동영상 끝에 도달하면 종료

    # YOLO로 공 객체 탐지
    results = model(frame)
    objects = results.pandas().xyxy[0]  # 탐지된 객체 확인
    
    for idx, obj in objects.iterrows():
        class_id = int(obj['class'])  # 클래스 ID
        label = obj['name']  # 클래스 이름
        confidence = obj['confidence']  # 신뢰도
        x1, y1, x2, y2 = int(obj['xmin']), int(obj['ymin']), int(obj['xmax']), int(obj['ymax'])  # bounding box 좌표

        # 바운딩 박스와 객체 정보 표시
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label} ({confidence:.2f})"
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)

        # 공 클래스일 때만 처리 (클래스 ID는 32로 가정)
        if label == 'ball':  # 클래스 이름이 'sports ball'일 때
            ball_width = x2 - x1

            # 초점 거리 계산 (한 번만 수행)
            if focalLength is None:
                focalLength = calculate_distance(ball_width, 1)

            # 현재 프레임에서 거리 계산
            distance = calculate_distance(ball_width, focalLength)

            # 궤적 추적
            current_time = time.time()
            ball_trajectories.append((distance, current_time))

            # 최근 두 지점 간 속도 계산
            if len(ball_trajectories) > 1:
                prev_distance, prev_time = ball_trajectories[-2]
                time_diff = current_time - prev_time
                distance_diff = distance - prev_distance
                if time_diff > 0:
                    speed = calculate_speed(distance_diff, time_diff)
                    print(f"Speed: {speed:.2f} km/h")

            # 프레임에 궤적 시각화
            for i in range(len(ball_trajectories) - 1):
                pt1 = (int(ball_trajectories[i][0] * 100), frame.shape[0] - 100)  # 시각화 위치 조정
                pt2 = (int(ball_trajectories[i + 1][0] * 100), frame.shape[0] - 100)
                cv2.line(frame, pt1, pt2, (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()