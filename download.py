from cotracker.predictor import CoTrackerPredictor,CoTrackerOnlinePredictor
import torch
import imageio.v3 as iio
from cotracker.utils.visualizer import Visualizer

device = 'cuda'
ckpt_path = "./checkpoints/scaled_online.pth"  # 换成你本地的权重路径

# 加载模型（不用写 mode / model 参数）
cotracker = CoTrackerOnlinePredictor(checkpoint=ckpt_path).to(device)

# 读视频
url = "apple.mp4"
frames = iio.imread(url, plugin="FFMPEG")
video = torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(device)  # B T C H W

# 预测
grid_size = 10
pred_tracks, pred_visibility = cotracker(video, grid_size=grid_size)

# 可视化
vis = Visualizer(save_dir="armor", pad_value=120, linewidth=3)
vis.visualize(video, pred_tracks, pred_visibility)

