import cv2
import numpy as np
def video_to_frames(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(f"{Axeloutput_dir}/frame_{frame_id:04d}.jpg", frame)
        frame_id += 1
    cap.release()


