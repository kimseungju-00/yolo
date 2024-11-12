import cv2
import torch
import time

# YOLO 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)

# 카메라 객체 생성
cap = cv2.VideoCapture(0)  # 0번 웹캠 사용

# 카메라 캘리브레이션 값 (예시)
KNOWN_DISTANCE = 10  # 포수와 투수 사이 거리 (미터)
KNOWN_WIDTH = 0.235  # 홈플레이트 폭 (미터)

# 거리 측정을 위한 함수
def calculate_distance(width, focalLength):
    return (KNOWN_WIDTH * focalLength) / width

# 구속 계산 함수
def calculate_speed(distance, time):
    speed = distance / time
    return speed * 3.6  # m/s to km/h

# 투구 구간 탐지 및 구속 계산
ball_trajectories = []
fps = cap.get(cv2.CAP_PROP_FPS)
focalLength = None  # 초점 거리 (추후 계산)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO로 야구공 객체 탐지
    results = model(frame)
    objects = results.pandas().xyxy[0]

    if len(objects) > 0:
        ball = objects.iloc[0]  # 첫 번째 탐지 객체
        class_id = int(ball.iloc[0])  # 클래스 ID (정수)
        
        # 야구공 클래스일 때만 처리 (클래스 ID는 0부터 시작하므로 0으로 가정)
        if class_id == 0:
            x1, y1, x2, y2 = [int(val) for val in ball.iloc[1:5]]  # bounding box 좌표 (정수)
            ball_width = x2 - x1

            # 초점 거리 계산 (한 번만 수행)
            if focalLength is None:
                focalLength = (ball_width * KNOWN_DISTANCE) / KNOWN_WIDTH

            # 현재 프레임에서 거리 계산
            distance = calculate_distance(ball_width, focalLength)

            # 궤적 추적
            prev_time = time.time()
            ball_trajectories.append((distance, prev_time))

            # 최근 두 지점 간 속도 계산
            if len(ball_trajectories) > 1:
                prev_distance, prev_time = ball_trajectories[-2]
                time_diff = prev_time - ball_trajectories[-1][1]
                distance_diff = distance - prev_distance

                if time_diff > 0:
                    speed = calculate_speed(distance_diff, time_diff)
                    print(f"Speed: {speed:.2f} km/h")

            # 프레임에 bounding box 및 궤적 시각화
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for i in range(len(ball_trajectories) - 1):
                pt1 = (int(ball_trajectories[i][0] * KNOWN_WIDTH / focalLength), frame.shape[0])
                pt2 = (int(ball_trajectories[i + 1][0] * KNOWN_WIDTH / focalLength), frame.shape[0])
                cv2.line(frame, pt1, pt2, (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()