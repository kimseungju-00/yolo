import cv2
import os
import numpy as np

def video_to_frames_optimized(video_path, output_folder, frame_interval=5, motion_threshold=1.5, resize_factor=0.5):
    # 비디오 파일 열기
    video_capture = cv2.VideoCapture(video_path)
    
    # 출력 폴더 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frame_count = 0
    saved_count = 0
    ret, prev_frame = video_capture.read()
    
    if not ret:
        print("비디오를 열 수 없습니다.")
        return
    
    # 초기 프레임 크기 조정 및 그레이스케일 변환
    prev_frame_resized = cv2.resize(prev_frame, (0, 0), fx=resize_factor, fy=resize_factor)
    prev_gray = cv2.cvtColor(prev_frame_resized, cv2.COLOR_BGR2GRAY)
    
    while True:
        # 다음 프레임 읽기
        ret, frame = video_capture.read()
        if not ret:
            break

        # 프레임 크기 조정 및 그레이스케일 변환
        frame_resized = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

        # 프레임 간 간격 적용
        if frame_count % frame_interval == 0:
            # Optical Flow 계산
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            
            # Optical Flow 크기 계산하여 움직임 감지
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mean_magnitude = np.mean(magnitude)

            # 움직임이 임계값을 초과하면 프레임 저장
            if mean_magnitude > motion_threshold:
                frame_filename = os.path.join(output_folder, f'frame_{saved_count:04d}.jpg')
                cv2.imwrite(frame_filename, frame)  # 원본 크기의 프레임 저장
                saved_count += 1

            # 이전 프레임 업데이트
            prev_gray = gray

        frame_count += 1

    # 비디오 캡처 해제
    video_capture.release()
    print("f'Total {saved_count} frames saved with interval {frame_interval} and resize factor {resize_factor}.")

# 예제 사용법
video_path = 'v1.mp4'
output_folder = 'output_frames_4'
video_to_frames_optimized(video_path, output_folder, frame_interval=5, motion_threshold=1.5, resize_factor=0.5)
