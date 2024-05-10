import time
import torch
from PIL import Image

# 加载YOLOv5模型
model = torch.hub.load('ultralytics/yolov5', 'runs/train/exp178/weights/best.pt')

# 准备测试图像
image = Image.open('VOCdevkit/VOC2007/JPEGImages/29.jpg')

# 进行推理
start_time = time.time()
results = model(image)
end_time = time.time()

# 计算推理时间
inference_time = end_time - start_time

# 计算FPS
fps = 1 / inference_time

print(f"推理时间：{inference_time:.2f}秒")
print(f"FPS：{fps:.2f}")