import cv2
import json

video_path = "your_path"
output_video_path = "your_output_path"
video_name = video_path.split("/")[-1]
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

annotations = []
current_frame_points = []
start_frame = None
clip_started = False
paused = False
out = None
zoom_scale = 1.0
zoom_center = None
current_frame = None  # 保存当前帧，用于暂停显示

def mouse_callback(event, x, y, flags, param):
    global current_frame_points, paused, out, zoom_scale, zoom_center, current_frame
    if clip_started and paused:
        # 根据缩放计算真实坐标
        if zoom_scale != 1.0 and zoom_center:
            cx, cy = zoom_center
            # 先计算相对于缩放区域的坐标
            w, h = current_frame.shape[1], current_frame.shape[0]
            w_half, h_half = int(w/(2*zoom_scale)), int(h/(2*zoom_scale))
            x1, y1 = max(cx - w_half, 0), max(cy - h_half, 0)
            # 点击坐标映射回原始帧
            x_real = int(x1 + x / zoom_scale)
            y_real = int(y1 + y / zoom_scale)
        else:
            x_real, y_real = x, y

        if event == cv2.EVENT_LBUTTONDOWN:
            if len(current_frame_points) < 8:
                current_frame_points.append({"id": len(current_frame_points), "x": float(x_real), "y": float(y_real)})
                print(f"标注点 {len(current_frame_points)}: ({x_real},{y_real})")

            if len(current_frame_points) == 8:
                frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                annotations.append({
                    "video_name": video_name,  # 用视频名代替 frame_index
                    "frame_index": frame_no,  # 如果还想保留帧号可以保留这一项
                    "points": current_frame_points.copy()
                })
                print(f"帧 {frame_no} 标注完成，已保存 8 个点")

                with open("annotations.json", "w") as f:
                    json.dump(annotations, f, indent=4)
                print("已自动保存 annotations.json")

                current_frame_points.clear()
                paused = False
                if out is None:
                    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                print("继续播放视频...")

        elif event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                zoom_scale *= 1.25
            else:
                zoom_scale /= 1.25
            zoom_center = (x, y)
            print(f"缩放: {zoom_scale:.2f} 以 ({x},{y}) 为中心")

cv2.namedWindow("VideoAnnotator")
cv2.setMouseCallback("VideoAnnotator", mouse_callback)

while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break
        current_frame = frame.copy()
    else:
        frame = current_frame.copy()  # 暂停时显示当前帧

    frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    display_frame = frame.copy()

    # 绘制已标注点
    for ann in annotations:
        if ann["frame_index"] == frame_no:
            for pt in ann["points"]:
                cv2.circle(display_frame, (int(pt["x"]), int(pt["y"])), 3, (0,0,255), -1)

    # 绘制当前标注点
    for pt in current_frame_points:
        cv2.circle(display_frame, (int(pt["x"]), int(pt["y"])), 3, (255,0,0), -1)

    # 放大显示
    if zoom_scale != 1.0 and zoom_center:
        cx, cy = zoom_center
        h, w = display_frame.shape[:2]
        w_half, h_half = int(w/(2*zoom_scale)), int(h/(2*zoom_scale))
        x1, y1 = max(cx - w_half, 0), max(cy - h_half, 0)
        x2, y2 = min(cx + w_half, w), min(cy + h_half, h)
        zoomed = display_frame[y1:y2, x1:x2]
        display_frame = cv2.resize(zoomed, (w, h))

    cv2.imshow("VideoAnnotator", display_frame)
    key = cv2.waitKey(30) & 0xFF

    if key == ord("q"):
        break
    elif key == ord("a") and not clip_started:
        start_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        clip_started = True
        paused = True  # 开始剪辑时暂停
        print(f"剪辑从帧 {start_frame} 开始，视频暂停等待标注")
    elif key == ord(" "):
        paused = not paused
        print("视频暂停" if paused else "视频播放")

    if clip_started and out and not paused:
        out.write(frame)

cap.release()
if out:
    out.release()
cv2.destroyAllWindows()

