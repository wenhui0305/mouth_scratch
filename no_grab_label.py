import pandas as pd
import numpy as np

# 视频的帧率
FRAMES_PER_SECOND = 10.011346192351331

# Excel文件路径
excel_file = r'D:\码仓\香港泰康诺生物科技有限公司\mouth_scratch\mouth_scratch.xlsx'  # 请替换为你的Excel文件路径

# 读取Excel文件
df = pd.read_excel(excel_file)

# 输出原始 DataFrame 的信息
print("原始 DataFrame 的信息:")
print(df.info())

# 输出筛选后的 DataFrame 的信息
print("筛选后的 DataFrame 的信息:")
print(df.info())

# 填充空值（假设你想要用0填充空值）
df.fillna(0, inplace=True)

# 筛选出非抓绕行为的数据
df = df[df['行为'] == 0.0].copy()

# 使用 ffill 方法填充分钟一列中的0值
df['分钟'] = df['分钟'].replace(0.0).fillna(method='ffill')
print(df)

# 将小时、分钟、起始秒转换为总起始秒
df['起始总秒数'] = df['小时'] * 3600 + df['分钟'] * 60 + df['起始秒']

# 将小时、分钟、终止秒转换为总终止秒
df['终止总秒数'] = df['小时'] * 3600 + df['分钟'] * 60 + df['终止秒']

# 将总起始秒数和总终止秒数转换为帧号
df['起始帧号'] = (df['起始总秒数'] * FRAMES_PER_SECOND).round().astype(int)
df['终止帧号'] = (df['终止总秒数'] * FRAMES_PER_SECOND).round().astype(int)

# 显示结果
print(df)

# 如果需要，可以将修改后的DataFrame保存回Excel文件
df.to_excel(r'D:\码仓\香港泰康诺生物科技有限公司\mouth_scratch\label_no_grab_with_frames.xlsx', index=False)