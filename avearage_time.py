import pandas as pd

# 读取Excel文件
file_path = 'D:\码仓\香港泰康诺生物科技有限公司\mouth_scratch\c1\副本2021-08-13(2)(1).xlsx'
df = pd.read_excel(file_path)

# 填充行为列中的空值为0
df['行为'] = df['行为'].fillna(0)

# 筛选行为为1的数据
behavior_1_df = df[df['行为'] == 1]

# 计算所有行为为1的起始秒和终止秒间隔的平均时间
time_intervals = behavior_1_df['终止秒'] - behavior_1_df['起始秒']
average_time_interval = time_intervals.mean()

# 输出结果
print(f"所有行为为1的起始秒和终止秒间隔的平均时间: {average_time_interval:.2f} 秒")

# 如果需要保存处理后的数据到新的Excel文件
output_file_path = 'processed_data.xlsx'
df.to_excel(output_file_path, index=False)
