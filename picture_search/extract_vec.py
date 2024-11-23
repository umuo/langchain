from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image

# 加载 CLIP 模型和处理器
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 加载并处理图像
image = Image.open("pictures/traffic-7033509_960_720.jpg")
inputs = processor(images=image, return_tensors="pt", padding=True)

# 获取图像特征向量
with torch.no_grad():
    image_features = model.get_image_features(**inputs)

vector = image_features.numpy().flatten()  # 将特征向量展平

print(vector)
