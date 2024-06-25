import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from openpyxl import Workbook
import matplotlib.pyplot as plt

# 加载保存的模型
model = tf.keras.models.load_model('D:\码仓\香港泰康诺生物科技有限公司\mouth_scratch\model_checkpoints\epoch-08-val_loss-0.1272.keras')
model.summary()
# 视频路径
video_path = r'D:\码仓\香港泰康诺生物科技有限公司\mouth_scratch\mouth\00\YJ_C48_2021-08-13_00_00.mp4'


# 分帧函数
def frame_extraction(video_path, chunk_duration=5):
    video = cv2.VideoCapture(video_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frames = []
    success, frame = video.read()
    while success:
        frames.append(frame)
        success, frame = video.read()

    chunk_size = fps * chunk_duration
    chunks = [frames[i:i + chunk_size] for i in range(0, len(frames), chunk_size)]
    return chunks, fps


# 预测函数
def predict_chunk(chunk):
    accuracies = []
    for frame in chunk:
        # 预处理图像
        frame_resized = cv2.resize(frame, (64, 64))
        frame_normalized = frame_resized / 255.0
        frame_expanded = np.expand_dims(frame_normalized, axis=0)

        # 模型预测
        prediction = model.predict(frame_expanded)
        accuracies.append(prediction[0][0])  # 假设模型输出是单个精度值

    return np.mean(accuracies)


# 主程序
chunks, fps = frame_extraction(video_path)
results = []

for i, chunk in enumerate(chunks):
    average_accuracy = predict_chunk(chunk)
    is_scratching = average_accuracy <= 0.8
    results.append((i, average_accuracy, is_scratching))

# 生成Excel表
df = pd.DataFrame(results, columns=['Block Index', 'Accuracy', 'Scratching Behavior'])
df.to_excel('video_analysis.xlsx', index=False)

# 可视化结果
plt.figure(figsize=(10, 6))
block_indices = df['Block Index']
accuracies = df['Accuracy']
plt.bar(block_indices, accuracies, color=['red' if acc >= 0.8 else 'blue' for acc in accuracies])
plt.axhline(y=0.8, color='green', linestyle='--', label='Threshold (0.8)')
plt.xlabel('Block Index')
plt.ylabel('Average Accuracy')
plt.title('Video Block Accuracy Visualization')
plt.legend()
plt.savefig('accuracy_visualization.png')
plt.show()
