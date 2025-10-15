# co_tracker---
利用了co_tracker开源的模型进行装甲板的视频数据集转变为多帧图像数据集
https://github.com/facebookresearch/co-tracker我们使用了这篇论文利用的co-tracker对我们的装甲板视频进行不经过训练的直接数据标注
环境配置和参数权重请查看上面的链接

1.运行video.py对录的装甲板视频进行标注，按a键开始标注，鼠标滚轮用来放大缩小，八个点标记完成后开始生成点的json文件，等待视频播放完后生成从开始标记帧之后的视频文件
补充：需自行修改文件输出输入路径
2.运行data_transition.py对刚才输出的json文件进行转换（模型需要的json文件和刚才那个的格式不一致）
3，运行python irobot_armour.py   --video_path ./your_video.mp4   --checkpoint ./checkpoints/scaled_offline.pth   --save_dir output   --points ./your_json.json
使用模型，需注意，由于模型较大，显存可能不足，所以代码中写的是强制cpu，如果有显存比较大的同学可以试用gpu跑推理
