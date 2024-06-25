import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# 定义加载帧图像并生成视频块的函数
def load_frames_as_video_blocks(directory, num_frames, image_height, image_width):
    video_blocks = []
    labels = []
    class_names = sorted(os.listdir(directory))
    for class_index, class_name in enumerate(class_names):
        class_dir = os.path.join(directory, class_name)
        frame_files = sorted(os.listdir(class_dir))
        frame_files = [os.path.join(class_dir, file) for file in frame_files if
                       file.endswith('.jpg') or file.endswith('.png')]

        for i in range(0, len(frame_files) - num_frames + 1, num_frames):
            video_block = []
            for j in range(num_frames):
                frame_path = frame_files[i + j]
                img = tf.keras.preprocessing.image.load_img(frame_path, target_size=(image_height, image_width))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                video_block.append(img_array)
            video_blocks.append(np.array(video_block))
            labels.append(class_index)
    return np.array(video_blocks), np.array(labels)

# 定义数据增强函数
def augment_data(video_blocks, labels):
    augmented_videos = []
    augmented_labels = []
    for i in range(len(video_blocks)):
        video = video_blocks[i]
        label = labels[i]
        augmented_videos.append(video)
        augmented_labels.append(label)

        # 数据增强
        flip_video = np.flip(video, axis=2)  # 水平翻转
        augmented_videos.append(flip_video)
        augmented_labels.append(label)
    return np.array(augmented_videos), np.array(augmented_labels)

# 定义数据集路径和参数
train_dir = r'D:\桌面\香港泰康诺生物科技有限公司\scratch\dataset\train'
validation_dir = r'D:\桌面\香港泰康诺生物科技有限公司\scratch\dataset\val'
test_dir = r'D:\桌面\香港泰康诺生物科技有限公司\scratch\dataset\test'
image_height = 64
image_width = 64
num_frames = 16
batch_size =4
epochs = 10

# 加载数据集
train_videos, train_labels = load_frames_as_video_blocks(train_dir, num_frames, image_height, image_width)
validation_videos, validation_labels = load_frames_as_video_blocks(validation_dir, num_frames, image_height,
                                                                   image_width)
test_videos, test_labels = load_frames_as_video_blocks(test_dir, num_frames, image_height, image_width)

# 数据增强
train_videos, train_labels = augment_data(train_videos, train_labels)

# 构建3D-CNN模型
def build_3d_cnn_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same'),
        #layers.BatchNormalization(),
        layers.MaxPooling3D((2, 2, 2)),
        layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same'),
        #layers.BatchNormalization(),
        layers.MaxPooling3D((2, 2, 2)),
        layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same'),
        #layers.BatchNormalization(),
        layers.MaxPooling3D((2, 2, 2)),
        layers.Flatten(),
        #layers.Dense(128, activation='relu'),
        #layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# 构建模型
input_shape = (num_frames, image_height, image_width, 3)
model = build_3d_cnn_model(input_shape)

# 编译模型
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 定义早停策略
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 训练模型
history = model.fit(train_videos, train_labels, epochs=epochs,
                    validation_data=(validation_videos, validation_labels),
                    batch_size=batch_size,
                    callbacks=[early_stopping])

# 评估模型
test_loss, test_acc = model.evaluate(test_videos, test_labels)
print('Test accuracy:', test_acc)

# 可视化训练过程
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()