import json

# 读取原 JSON 文件
#通过video出来的json文件的路径
with open("your_json.json", "r") as f:
    data = json.load(f)

# 假设你只想转换第一帧的点
first_frame = data[0]  # 如果你想处理特定帧，可以按索引调整
converted = [[pt["x"], pt["y"]] for pt in first_frame["points"]]

# 保存到新文件
with open("your_new_json.json", "w") as f:
    json.dump(converted, f, indent=4)

#print("转换完成，已保存为 annotations_converted.json")

