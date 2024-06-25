import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

# 文件夹路径
grab_folder = r'D:\码仓\香港泰康诺生物科技有限公司\mouth_scratch\frames_scratch'
no_grab_folder = r'D:\码仓\香港泰康诺生物科技有限公司\mouth_scratch\frames_no_scratch'

# 获取所有图像文件名
grab_images = os.listdir(grab_folder)
no_grab_images = os.listdir(no_grab_folder)

# 抓挠行为总数
num_grab_frames = len(grab_images)
# 计算抓挠行为按比例划分的帧数
grab_train_num = int(num_grab_frames * 0.4)
grab_val_num = int(num_grab_frames * 0.2)
grab_test_num = num_grab_frames - grab_train_num - grab_val_num

# 非抓挠行为总数
num_no_grab_frames = (len(grab_images))*5
print(num_no_grab_frames)
# 计算非抓挠行为按比例划分的帧数
no_grab_train_num = int(num_no_grab_frames * 0.4)
no_grab_val_num = int(num_no_grab_frames * 0.2)
print(no_grab_val_num)
no_grab_test_num = num_no_grab_frames - no_grab_train_num - no_grab_val_num

# 划分抓挠行为图像
grab_train, grab_temp = train_test_split(grab_images, train_size=grab_train_num, test_size=grab_val_num + grab_test_num, random_state=42)
grab_val, grab_test = train_test_split(grab_temp, test_size=grab_test_num, random_state=43)

# 划分非抓挠行为图像
no_grab_train, no_grab_temp = train_test_split(no_grab_images, train_size=no_grab_train_num,test_size=no_grab_val_num + no_grab_test_num, random_state=42)
no_grab_val, no_grab_test = train_test_split(no_grab_temp, test_size=no_grab_test_num, random_state=43)

# 创建目标文件夹
def create_folders(base_dir, sets):
    for set_name in sets:
        os.makedirs(os.path.join(base_dir, set_name, 'grab'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, set_name, 'no_grab'), exist_ok=True)

# 复制图像到目标文件夹
def copy_images(image_list, source_folder, target_folder):
    for image in image_list:
        shutil.copy(os.path.join(source_folder, image), os.path.join(target_folder, image))

# 设置数据集目标文件夹
base_dir = 'dataset'
sets = ['train', 'val', 'test']
create_folders(base_dir, sets)

# 复制抓挠行为图像
for set_name, grab_set in zip(sets, [grab_train, grab_val, grab_test]):
    copy_images(grab_set, grab_folder, os.path.join(base_dir, set_name, 'grab'))

# 复制非抓挠行为图像
for set_name, no_grab_set in zip(sets, [no_grab_train, no_grab_val, no_grab_test]):
    copy_images(no_grab_set, no_grab_folder, os.path.join(base_dir, set_name, 'no_grab'))

# 输出数据集大小
print(f"训练集抓挠行为图像数量: {len(grab_train)}")
print(f"训练集非抓挠行为图像数量: {len(no_grab_train)}")
print(f"验证集抓挠行为图像数量: {len(grab_val)}")
print(f"验证集非抓挠行为图像数量: {len(no_grab_val)}")
print(f"测试集抓挠行为图像数量: {len(grab_test)}")
print(f"测试集非抓挠行为图像数量: {len(no_grab_test)}")
