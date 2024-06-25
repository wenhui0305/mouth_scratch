import os
import pandas as pd
import shutil

# Excel文件路径
excel_file = r'D:\桌面\香港泰康诺生物科技有限公司\scratch\label_with_frames_no_scrach.xlsx'
# 帧图像的文件夹路径
frame_images_folder = r'D:\桌面\香港泰康诺生物科技有限公司\scratch\frame'
# 目标文件夹路径，将帧图像复制到这里
output_folder = r'D:\桌面\香港泰康诺生物科技有限公司\scratch\frames_no_scratch'

# 假设帧号的上限是36018
max_frame_number = 36018

# 读取Excel文件
df = pd.read_excel(excel_file)

# 创建一个集合来保存所有列出的帧号
listed_frames = set()
for index, row in df.iterrows():
    start_frame = row['起始帧号']
    end_frame = row['终止帧号']

    # 检查起始帧号和终止帧号是否有效（非空且为正整数）
    if pd.notnull(start_frame) and pd.notnull(end_frame) and start_frame <= end_frame:
        listed_frames.update(range(start_frame, end_frame + 1))  # 更新集合，添加范围内的帧号

# 创建一个集合来保存所有可能的帧号
possible_frames = set(range(1, max_frame_number + 1))

# 找出未列出的帧号
frames_to_extract = possible_frames - listed_frames

# 确保目标文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历未列出的帧号，并从文件系统中提取相应的帧图像
for frame_number in frames_to_extract:
    frame_filename = f"{frame_number}.jpg"
    frame_path = os.path.join(frame_images_folder, frame_filename)
    output_path = os.path.join(output_folder, frame_filename)

    # 检查文件是否存在，如果存在则复制
    if os.path.exists(frame_path):
        shutil.copy(frame_path, output_path)
        print(f"Copied frame: {frame_path} to {output_path}")
    else:
        print(f"Frame with number {frame_number} not found: {frame_path}")

print("Frame extraction complete for frames without specified range.")