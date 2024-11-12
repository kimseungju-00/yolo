import cv2
import os

def video_to_frames(video_path, output_folder):
    # 비디오 파일을 열기
    video_capture = cv2.VideoCapture(video_path)
    
    # 출력 폴더가 없으면 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frame_count = 0
    while True:
        # 프레임 읽기
        ret, frame = video_capture.read()
        
        # 더 이상 프레임이 없으면 종료
        if not ret:
            break
        
        # 프레임 저장
        frame_filename = os.path.join(output_folder, f'frame_{frame_count:04d}.jpg')
        cv2.imwrite(frame_filename, frame)
        
        frame_count += 1

    # 비디오 캡처 해제
    video_capture.release()
    print(f'Total {frame_count} frames saved.')

# 예제 사용법
video_path = 'v4.mp4'
output_folder = 'output_frames_6'
video_to_frames(video_path, output_folder)
