import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

# 文件夹路径
grab_folder = r'D:\桌面\香港泰康诺生物科技有限公司\scratch\frames_scratch'
no_grab_folder = r'D:\桌面\香港泰康诺生物科技有限公司\scratch\frames_no_scratch'

# 获取所有图像文件名
grab_images = os.listdir(grab_folder)
no_grab_images = os.listdir(no_grab_folder)

# 划分抓绕行为图像
grab_train, grab_temp = train_test_split(grab_images, test_size=0.20, random_state=42)
grab_val, grab_test = train_test_split(grab_temp, test_size=0.50, random_state=42)

# 划分不抓绕行为图像
no_grab_train, no_grab_temp = train_test_split(no_grab_images, test_size=0.20, random_state=42)
no_grab_val, no_grab_test = train_test_split(no_grab_temp, test_size=0.50, random_state=42)

# 平衡数据集（如果需要）
# 例如：欠采样不抓绕行为图像使其数量与抓绕行为图像数量相等
balanced_no_grab_train = np.random.choice(no_grab_train, size=len(grab_train), replace=False)
balanced_no_grab_val = np.random.choice(no_grab_val, size=len(grab_val), replace=False)
balanced_no_grab_test = np.random.choice(no_grab_test, size=len(grab_test), replace=False)

# 创建目标文件夹
os.makedirs('dataset/train/grab', exist_ok=True)
os.makedirs('dataset/train/no_grab', exist_ok=True)
os.makedirs('dataset/val/grab', exist_ok=True)
os.makedirs('dataset/val/no_grab', exist_ok=True)
os.makedirs('dataset/test/grab', exist_ok=True)
os.makedirs('dataset/test/no_grab', exist_ok=True)

# 函数：复制图像到目标文件夹
def copy_images(image_list, source_folder, target_folder):
    for image in image_list:
        shutil.copy(os.path.join(source_folder, image), target_folder)

# 复制抓绕行为图像
copy_images(grab_train, grab_folder, 'dataset/train/grab')
copy_images(grab_val, grab_folder, 'dataset/val/grab')
copy_images(grab_test, grab_folder, 'dataset/test/grab')

# 复制不抓绕行为图像
copy_images(balanced_no_grab_train, no_grab_folder, 'dataset/train/no_grab')
copy_images(balanced_no_grab_val, no_grab_folder, 'dataset/val/no_grab')
copy_images(balanced_no_grab_test, no_grab_folder, 'dataset/test/no_grab')

# 输出数据集大小
print(f"训练集抓绕行为图像数量: {len(grab_train)}")
print(f"训练集不抓绕行为图像数量: {len(balanced_no_grab_train)}")
print(f"验证集抓绕行为图像数量: {len(grab_val)}")
print(f"验证集不抓绕行为图像数量: {len(balanced_no_grab_val)}")
print(f"测试集抓绕行为图像数量: {len(grab_test)}")
print(f"测试集不抓绕行为图像数量: {len(balanced_no_grab_test)}")
