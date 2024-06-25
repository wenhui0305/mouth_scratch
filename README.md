# mouth_scratch
老鼠抓挠算法--相关代码（包括划分数据和数据建模）
extract_frame.py-----对视频分帧处理
grab_label.py-----提取抓挠行为标签          生成表格label_grab_frames
no_grab_label.py-----提取非抓挠行为标签       生成表格label_no_grab_with_frames
scratching.py-----将提取的抓挠行为标签与视频帧一一对应提取保存文件夹frames_scratch
no_scratching.py-----将提取的非抓挠行为标签与视频帧一一对应提取保存文件夹frames_scratch
data set partition.py-----将抓挠文件夹与非抓挠文件夹划分数据集，训练、验证、测试4：2：4
2.py-----构建2D-CNN模型并可视化训练过程（目前有bug)
ceshi.py-----可以将随机视频放入保存好的模型进行测试，并生成一个excel表查看是否抓挠
average_time.py-----提取所有抓挠行为标签发生起始秒和终止秒的平均间隔，确定视频块的大小



