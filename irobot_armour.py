import os
import torch
import argparse
import json
import numpy as np
import cv2
from PIL import Image
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerPredictor

#DEFAULT_DEVICE = (
    #"cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
#)
DEFAULT_DEVICE="cpu"

def parse_args():
    parser = argparse.ArgumentParser(description="CoTracker Demo 参数配置")

    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="输入视频的路径，例如：/workspace/data/test.mp4"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/scaled_offline.pth",
        help="模型权重路径（默认 ./checkpoints/scaled_offline.pth）"
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="output_rm",
        help="输出保存路径（默认 output_rm）"
    )

    parser.add_argument(
        "--points",
        type=str,
        default=None,
        help="初始跟踪点坐标，可以是 JSON 文件路径或直接输入 JSON 字符串，例如 --points '[ [556,256], [553,317] ]'"
    )

    return parser.parse_args()


def load_points(points_arg):
    if points_arg is None:
        raise ValueError("❌ 请使用 --points 参数提供跟踪点坐标！")

    # 如果传入的是文件路径
    if os.path.exists(points_arg):
        with open(points_arg, "r") as f:
            pts_xy = json.load(f)
    else:
        # 否则直接解析字符串形式的 JSON
        pts_xy = json.loads(points_arg)

    return pts_xy

def main_custom():
    args = parse_args()
    video_path = args.video_path
    checkpoint = args.checkpoint
    save_dir = args.save_dir
    pts_xy = load_points(args.points)
    #这里全部都是命令行输入的，如果有同学不太熟练的话，可以把这个地方的参数直接写死在代码里
    #例如 video_path = "./test.mp4"(相对路径)
    print(f"✅ 视频路径: {video_path}")
    print(f"✅ 权重路径: {checkpoint}")
    print(f"✅ 输出目录: {save_dir}")
    print(f"✅ 跟踪点: {pts_xy}")
    use_v2_model = False
    offline = True

    # 创建输出文件夹
    os.makedirs(os.path.join(save_dir, "frames"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "json"), exist_ok=True)

    # 读取视频
    video = read_video_from_path(video_path)  # (T, H, W, C)
    video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
    print("video.shape =", video_tensor.shape)

    B, T, C, H, W = video_tensor.shape

    # 加载模型
    if checkpoint is not None:
        if use_v2_model:
            model = CoTrackerPredictor(checkpoint=checkpoint, v2=use_v2_model)
            print("Using CoTracker v2 model")
        else:
            model = CoTrackerPredictor(
                checkpoint=checkpoint,
                v2=use_v2_model,
                offline=offline,
                window_len=60,
            )
            print(f"Using CoTracker {'offline' if offline else 'online'} model")
    else:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")

    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device="cpu"
    #如果出现显存不够的情况再跟我说，这里默认大家的电脑显存是够的
    model = model.to(device)
    video_tensor = video_tensor.to(device)

    # 初始点 如果使用命令行输入的话这里就注释掉了
    #pts_xy = [
        #[556, 256],
        #[553, 317],
        #[407, 312],
        #[408, 251],
        #[515, 236],
        #[517, 338],
        #[443, 335],
        #[446, 259],
    #]

    t0 = 0
    queries_np = np.array([[[t0, x, y] for (x, y) in pts_xy]], dtype=np.float32)
    queries_float = torch.from_numpy(queries_np).to(device)

    # 调用模型
    with torch.no_grad():
        pred_tracks, pred_visibility = model(
            video_tensor,
            queries=queries_float,
            grid_query_frame=0,
            backward_tracking=False,
        )

    print("Tracking computed.")
    pred_tracks = pred_tracks.cpu().numpy()[0]  # remove batch dim
    print("pred_tracks.shape =", pred_tracks.shape)

    # -------- ✅ 自动检测并修正维度顺序 --------
    # 模型可能返回 (T, N, 2) 或 (N, T, 2)，我们保证最后是 (N, T, 2)
    if pred_tracks.shape[0] == T and pred_tracks.shape[1] != T:
        # 当前是 (T, N, 2)
        pred_tracks = np.transpose(pred_tracks, (1, 0, 2))
        print("→ transposed pred_tracks to (N, T, 2)")
    elif pred_tracks.shape[1] == T:
        print("→ pred_tracks already (N, T, 2)")
    else:
        raise ValueError(f"Unexpected shape {pred_tracks.shape}, cannot match T={T}")
    # ----------------------------------------

    print("final pred_tracks.shape =", pred_tracks.shape)

    # 遍历每一帧
    for t in range(T):
        frame = video[t].copy()  # numpy (H, W, C)
        frame_points = []

        for i, (x, y) in enumerate(pred_tracks[:, t, :]):
            frame_points.append({"id": int(i), "x": float(x), "y": float(y)})

            # 绘制点
            if 0 <= int(x) < frame.shape[1] and 0 <= int(y) < frame.shape[0]:
                cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)
                cv2.putText(frame, f"{i}", (int(x) + 5, int(y) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 保存图像
        frame_path = os.path.join(save_dir, "frames", f"frame_{t:04d}.png")
        cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # 保存 JSON
        json_data = {
            "frame_index": int(t),
            "points": frame_points
        }
        json_path = os.path.join(save_dir, "json", f"frame_{t:04d}.json")
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=4)

    print(f"✅ All frames and JSONs saved to {save_dir}/frames and {save_dir}/json")

if __name__ == "__main__":
    main_custom()