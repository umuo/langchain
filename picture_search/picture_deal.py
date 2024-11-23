import os

pic_path = '/home/gitsilence/Pictures/test'

from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import numpy as np
from paddleocr import PaddleOCR
from opensearchpy import OpenSearch

ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # need to run only once to download and load model into memory
# 加载 CLIP 模型和处理器
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 初始化 OpenSearch 客户端
client = OpenSearch(
    hosts=[{'host': '127.0.0.1', 'port': 9200}],
    http_auth=('admin', 'gggMz88.'),  # 默认用户名和密码，如果已修改需替换
    use_ssl=True,
    verify_certs=False,
    ssl_show_warn=False
)

def get_img_list() -> list:
    img_list = []
    files = os.listdir(pic_path)
    for file in files:
        if file.endswith("jpg") or file.endswith("png"):
            print(file)
            img_list.append(file)
    return img_list

def extract_vec_text(img: str):
    # 加载并处理图像
    image = Image.open(os.path.join(pic_path, img))
    inputs = processor(images=image, return_tensors="pt", padding=True)

    # 获取图像特征向量
    with torch.no_grad():
        vector = model.get_image_features(**inputs).squeeze().numpy()

    # vector = image_features.numpy().flatten()  # 将特征向量展平
    vector /= np.linalg.norm(vector)  # 归一化
    vector = vector.astype(float).tolist()  # 转换为浮点数列表

    # OCR 文本提取
    ocr_result = ocr.ocr(os.path.join(pic_path, img), cls=True)
    if ocr_result is None or ocr_result[0] is None:
        return vector, ""
    result = [line[1][0] for line in ocr_result[0]]
    ocr_text = "" if result is None else "".join(result)  # 提取文字内容
    return vector, ocr_text

def upload_to_opensearch(image_path, vector, ocr_text):
    """将特征向量和 OCR 文本写入 OpenSearch"""
    doc = {
        "image_vector": vector,  # 转为列表存储
        "ocr_text": ocr_text,
        "image_path": "http://127.0.0.1:5500/" + image_path
    }
    response = client.index(index="image-index", body=doc)
    print("Document indexed:", response["_id"])
    print(client, doc)

if __name__ == '__main__':
    imgs = get_img_list()
    for img in imgs:
        vex, text = extract_vec_text(img)
        print(text)
        upload_to_opensearch(img, vex, text)
        print("--------------------------")
    pass
