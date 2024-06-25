import cv2
import os
import glob

# 视频文件夹路径
video_folder_path = r"D:\桌面\香港泰康诺生物科技有限公司\scratch\2021-08-13\00"

# 创建保存帧图像的目录
output_dir ="frame"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 提取所有 MP4 文件的路径
video_files = glob.glob(os.path.join(video_folder_path, "*.mp4"))

frame_count = 1

# 遍历所有视频文件并提取帧
for video_file in video_files:
    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_file}")
        continue

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 保存帧图像到文件
        frame_filename = os.path.join(output_dir, f"{frame_count}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()

print("Frames extracted and saved successfully.")