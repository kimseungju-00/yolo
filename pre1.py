import cv2
import numpy as np
import os

# 출력 폴더 생성
output_folder = "interpolated_frames"
os.makedirs(output_folder, exist_ok=True)

# 동영상 파일 열기
cap = cv2.VideoCapture('v3.mp4')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
output = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 60, (frame_width, frame_height))

# 첫 번째 프레임 읽기
ret, prev_frame = cap.read()
frame_count = 0  # 프레임 카운트 초기화
if not ret:
    print("Failed to read video.")
    cap.release()
    output.release()
    exit()

# 다음 프레임 읽기 및 보간 수행
while cap.isOpened():
    ret, next_frame = cap.read()
    if not ret:
        break

    # 그레이스케일로 변환 (Optical Flow 계산에 사용)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    # Optical Flow 계산
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # 중간 프레임 생성
    alpha = 0.5  # 중간 비율 조정 (0.5는 정확히 중간 프레임)
    mid_frame = np.zeros_like(prev_frame)
    
    # Optical Flow를 이용해 중간 프레임 보간
    for y in range(frame_height):
        for x in range(frame_width):
            dx, dy = flow[y, x].astype(int)
            src_x = min(max(x + int(dx * alpha), 0), frame_width - 1)
            src_y = min(max(y + int(dy * alpha), 0), frame_height - 1)
            mid_frame[y, x] = (1 - alpha) * prev_frame[y, x] + alpha * next_frame[src_y, src_x]

    # 결과 프레임 저장
    output.write(prev_frame)  # 원본 프레임
    output.write(mid_frame)   # 보간된 중간 프레임
    
    # 보간된 중간 프레임을 이미지 파일로 저장
    mid_frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}_interpolated.jpg")
    cv2.imwrite(mid_frame_filename, mid_frame)
    frame_count += 1

    prev_frame = next_frame   # 다음 프레임 준비

cap.release()
output.release()
cv2.destroyAllWindows()
