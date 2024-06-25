import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 加载训练好的模型
model = load_model('final_model1.keras')

# 设置测试集路径
test_dir = 'dataset/test/'

# 图像生成器用于数据标准化
test_datagen = ImageDataGenerator(rescale=1.0/255)

# 生成测试数据
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# 使用模型进行预测
predictions = model.predict(test_generator)
predicted_classes = np.where(predictions > 0.5, 1, 0)

# 获取真实的标签
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# 打印分类报告
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)

# 打印混淆矩阵
conf_matrix = confusion_matrix(true_classes, predicted_classes)
print(conf_matrix)

# 可视化混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
