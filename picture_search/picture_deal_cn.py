import os

pic_path = '/home/gitsilence/Pictures/test'

import torch
from PIL import Image
import numpy as np
from paddleocr import PaddleOCR
from opensearchpy import OpenSearch

import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
print("Available models:", available_models())
# Available models: ['ViT-B-16', 'ViT-L-14', 'ViT-L-14-336', 'ViT-H-14', 'RN50']

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = load_from_name("ViT-B-16", device=device, download_root='./')
model.eval()

ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # need to run only once to download and load model into memory

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
    inputs = preprocess(image).unsqueeze(0).to(device)
    image_features = model.encode_image(inputs)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    # # 将特征转换为浮点数列表，断开计算图 将特征转换为浮点数列表
    vector = image_features.detach().squeeze().cpu().numpy().astype(float).tolist()
    # text = clip.tokenize(["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"]).to(device)
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
        print(vex)
        upload_to_opensearch(img, vex, text)
        print("--------------------------")
    pass
