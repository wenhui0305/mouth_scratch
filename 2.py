import os
import numpy as np
import matplotlib.pyplot as plt
from keras import regularizers
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import roc_curve, auc, classification_report, precision_recall_curve
import re

# 定义数据集路径和参数
train_dir = r'D:\码仓\香港泰康诺生物科技有限公司\mouth_scratch\dataset\train'  # 训练集路径
validation_dir = r'D:\码仓\香港泰康诺生物科技有限公司\mouth_scratch\dataset\val'  # 验证集路径
test_dir = r'D:\码仓\香港泰康诺生物科技有限公司\mouth_scratch\dataset\test'  # 测试集路径
image_height = 64  # 缩小图像尺寸
image_width = 64   # 缩小图像尺寸
batch_size = 16     # 减小批量大小
epochs = 10  # 训练的轮数

# 加载和准备数据集
train_datagen = ImageDataGenerator(rescale=1./255)  # 训练数据生成器，图像像素值缩放到[0, 1]
validation_datagen = ImageDataGenerator(rescale=1./255)  # 验证数据生成器
test_datagen = ImageDataGenerator(rescale=1./255)  # 测试数据生成器

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='binary'
)  # 生成训练数据

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='binary'
)  # 生成验证数据

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='binary'
)  # 生成测试数据

# 构建卷积神经网络模型
def build_model(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),  # 第一个卷积层
        layers.MaxPooling2D((2, 2)),  # 第一个最大池化层
        layers.Conv2D(64, (3, 3), activation='relu'),  # 第二个卷积层
        layers.MaxPooling2D((2, 2)),  # 第二个最大池化层
        layers.Conv2D(128, (3, 3), activation='relu'),  # 第三个卷积层
        layers.MaxPooling2D((2, 2)),  # 第三个最大池化层
        layers.Flatten(),  # 展平层
        layers.Dense(128, activation='relu'),  # 全连接层(解决过拟合）
        layers.Dropout(0.5),  # Dropout层
        layers.Dense(1, activation='sigmoid')  # 输出层
    ])
    return model

# 构建模型
input_shape = (image_height, image_width, 3)  # 输入图像形状
model = build_model(input_shape)

# 编译模型
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])  # 使用adam优化器和二元交叉熵损失函数

# 创建一个目录来保存模型
checkpoint_dir = 'model_checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)  # 创建目录以保存模型检查点

# # 定义早停策略和检查点回调
# early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)  # 早停策略
checkpoint_callback = ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, 'epoch-{epoch:02d}-val_accuracy-{val_accuracy:.4f}.keras'),  # 模型文件名包含验证准确性
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=False,
    mode='max',
    save_freq='epoch'
)  # 模型检查点回调

# 训练模型，同时保存历史记录
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[checkpoint_callback]
    # callbacks=[early_stopping, checkpoint_callback]
)  # 训练模型

# 可视化准确率和损失曲线
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(10, 8))
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
    plt.savefig('training_validation_curves.png')
    plt.show()

plot_history(history)  # 绘制训练和验证曲线

# 找出最优模型文件
checkpoint_files = os.listdir(checkpoint_dir)  # 列出检查点目录下的所有文件
print(f"检查点目录中的文件: {checkpoint_files}")

# 匹配文件名中的验证准确性
accuracy_pattern = re.compile(r'val_accuracy-(\d+\.\d+)\.keras')
accuracy_values = [(file, float(accuracy_pattern.search(file).group(1))) for file in checkpoint_files if accuracy_pattern.search(file)]
print(f"匹配到的文件和验证准确性: {accuracy_values}")

# 如果存在匹配到的模型文件
if accuracy_values:
    best_model_file = max(accuracy_values, key=lambda x: x[1])[0]  # 找到验证准确性最大的模型文件
    print(f"最优模型文件为: {best_model_file}")

    for model_file, _ in accuracy_values:
        # 加载模型权重
        model.load_weights(os.path.join(checkpoint_dir, model_file))

        # 评估模型在测试集上的表现
        print("最终测试评估：")
        test_loss, test_acc = model.evaluate(test_generator)  # 在测试集上评估模型
        print(f"val Loss: {test_loss}")
        print('val accuracy:', test_acc)

        # 预测测试数据
        y_test_predictions = model.predict(test_generator, verbose=1)  # 预测测试集数据
        test_predictions = np.where(y_test_predictions > 0.5, 1, 0)  # 将概率转换为类别标签

        # 获取真实标签
        true_labels = test_generator.classes

        # 计算ROC曲线
        fpr, tpr, _ = roc_curve(true_labels, y_test_predictions)
        roc_auc = auc(fpr, tpr)

        # 绘制ROC曲线
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic Curve')
        plt.legend(loc="lower right")
        plt.show()
else:
    print("未找到任何匹配的模型文件")
#
# # 如果存在匹配到的模型文件
# if accuracy_values:
#     best_model_file = max(accuracy_values, key=lambda x: x[1])[0]  # 找到验证准确性最大的模型文件
#     print(f"最优模型文件为: {best_model_file}")
#
#     # 加载最优模型权重
#     model.load_weights(os.path.join(checkpoint_dir, best_model_file))  # 加载最优模型的权重
#
#     # 评估模型在测试集上的表现
#     print("最终测试评估：")
#     test_loss, test_acc = model.evaluate(test_generator)  # 在测试集上评估模型
#     print(f"Test Loss: {test_loss}")
#     print('Test accuracy:', test_acc)
#
#     # 预测测试数据
#     y_test_predictions = model.predict(test_generator, verbose=1)  # 预测测试集数据
#     test_predictions = np.where(y_test_predictions > 0.5, 1, 0)  # 将概率转换为类别标签
#
#     # 获取真实标签
#     true_labels = test_generator.classes  # 测试集的真实标签
#
#     # 打印分类报告
#     print("分类报告:\n", classification_report(true_labels, test_predictions, target_names=['no_grab', 'grab']))
#
#     # 计算并绘制ROC和PR曲线
#     fpr, tpr, roc_thresholds = roc_curve(true_labels, y_test_predictions)  # 计算ROC曲线
#     roc_auc = auc(fpr, tpr)  # 计算AUC值
#
#     precision, recall, pr_thresholds = precision_recall_curve(true_labels, y_test_predictions)  # 计算PR曲线
#     pr_auc = auc(recall, precision)  # 计算AUC值
#
#     plt.figure(figsize=(12, 6))
#
#     # 绘制ROC曲线
#     plt.subplot(1, 2, 1)
#     plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
#     plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver Operating Characteristic (ROC) Curve')
#     plt.legend()
#
#     # 绘制PR曲线
#     plt.subplot(1, 2, 2)
#     plt.plot(recall, precision, color='blue', label=f'PR curve (area = {pr_auc:.2f})')
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.title('Precision-Recall (PR) Curve')
#     plt.legend()
#
#     plt.tight_layout()
#     plt.show()
#
#     # 找出最佳阈值
#     f1_scores = 2 * (precision * recall) / (precision + recall)  # 计算F1分数
#     best_threshold_index = np.argmax(f1_scores)  # 找到F1分数最大的阈值索引
#     best_threshold = pr_thresholds[best_threshold_index]  # 对应的最佳阈值
#
#     print(f'Best threshold: {best_threshold}')
#     print(f'Best F1 score: {f1_scores[best_threshold_index]}')
# else:
#     print("未找到任何匹配的模型文件")
