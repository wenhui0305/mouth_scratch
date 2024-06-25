import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report

# 定义数据集路径和参数
train_dir = r'D:\桌面\香港泰康诺生物科技有限公司\scratch\dataset\train'
validation_dir = r'D:\桌面\香港泰康诺生物科技有限公司\scratch\dataset\val'
test_dir = r'D:\桌面\香港泰康诺生物科技有限公司\scratch\dataset\test'
image_height = 64  # 缩小图像尺寸
image_width = 64   # 缩小图像尺寸
batch_size = 16     # 减小批量大小
epochs = 10

# 加载和准备数据集
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='binary'
)

# 构建CNN模型
def build_model(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# 构建模型
input_shape = (image_height, image_width, 3)
model = build_model(input_shape)

# 编译模型
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 定义早停策略
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# 训练模型
history = model.fit(train_generator, epochs=epochs,
                    validation_data=validation_generator,
                    callbacks=[early_stopping])

# 评估模型
test_loss, test_acc = model.evaluate(test_generator)
print('Test accuracy:', test_acc)

# 对测试数据进行预测
y_pred_prob = model.predict(test_generator)
y_pred = np.where(y_pred_prob > 0.5, 1, 0)  # 将概率转换为类别标签

# 获取实际标签
test_labels = test_generator.classes

'''''
# 替换类别标签
class_names = ["no_grab", "grab"]
test_labels = [class_names[label] for label in test_labels]
y_pred = [class_names[label[0]] for label in y_pred]

# 打印分类报告
print(classification_report(test_labels, y_pred))
'''

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
