import os
import shutil
import pandas as pd

excel_file = r'D:\桌面\香港泰康诺生物科技有限公司\scratch\label_with_frames.xlsx'
sheet_name = 'Sheet1'

# 读取Excel数据
df = pd.read_excel(excel_file, sheet_name=sheet_name)

# 帧图像的文件夹路径
frame_images_folder = r'D:\桌面\香港泰康诺生物科技有限公司\scratch\frame'
# 目标文件夹路径，将帧图像复制到这里
output_folder = r'D:\桌面\香港泰康诺生物科技有限公司\scratch\output_frames'

# 确保目标文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历Excel中的每一行
for index, row in df.iterrows():
    start_frame = int(row['起始帧号'])
    end_frame = int(row['终止帧号'])

    # 遍历并复制帧图像
    for frame_num in range(start_frame, end_frame + 1):
        # 假设帧图像的文件名是纯数字，如 "110"
        frame_filename = f"{frame_num}.jpg"  # 假设文件扩展名是.jpg
        frame_path = os.path.join(frame_images_folder, frame_filename)

        # 检查文件是否存在
        if os.path.exists(frame_path):
            # 复制帧图像到目标文件夹
            output_path = os.path.join(output_folder, frame_filename)
            shutil.copy(frame_path, output_path)
            print(f"Copied frame: {frame_path} to {output_path}")
        else:
            print(f"Frame {frame_num} not found: {frame_path}")

print("Frame extraction complete.")