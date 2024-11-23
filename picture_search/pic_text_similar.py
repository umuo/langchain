"""
图片和文本相似度比较
"""
import torch
from PIL import Image

import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
print("Available models:", available_models())
# Available models: ['ViT-B-16', 'ViT-L-14', 'ViT-L-14-336', 'ViT-H-14', 'RN50']

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = load_from_name("ViT-B-16", device=device, download_root='./')
model.eval()
labels = ["两个人骑摩托车", "街道上的人和摩托车", "两个人正在街上驾驶摩托车", "两个孙悟空"]


image = preprocess(Image.open("pictures/traffic-7033509_960_720.jpg")).unsqueeze(0).to(device)
text = clip.tokenize(labels).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    # 对特征进行归一化，请使用归一化后的图文特征用于下游任务
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    logits_per_image, logits_per_text = model.get_similarity(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# 打印每个标签及其对应的概率
for label, prob in zip(labels, probs[0]):
    print(f"{label}: {prob:.4f}")
