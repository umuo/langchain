import os
from picture_search.search_test_cn import preprocess

pic_path = '/home/gitsilence/Pictures/test'

import torch
from PIL import Image
from paddleocr import PaddleOCR
from opensearchpy import OpenSearch
import onnxruntime

from cn_clip.clip.utils import image_transform, _MODEL_INFO

# Available models: ['ViT-B-16', 'ViT-L-14', 'ViT-L-14-336', 'ViT-H-14', 'RN50']


# 载入ONNX图形侧模型
img_sess_option = onnxruntime.SessionOptions()
img_run_options = onnxruntime.RunOptions()
img_run_options.log_severity_level = 2
img_onnx_model_path = "/media/gitsilence/develop/PythonProject/Chinese-CLIP/cn_clip/deploy/vit-b-16.img.fp32.onnx"
img_session = onnxruntime.InferenceSession(
    img_onnx_model_path,
    sess_options=img_sess_option,
    providers=["CUDAExecutionProvider"]
)
model_arch = "ViT-B-16"
preprocess = image_transform(_MODEL_INFO[model_arch]['input_resolution'])

ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # need to run only once to download and load model into memory

# 初始化 OpenSearch 客户端
client = OpenSearch(
    hosts=[{'host': '127.0.0.1', 'port': 9200}],
    http_auth=('admin', 'gggMz123...'),  # 默认用户名和密码，如果已修改需替换
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
    # 预处理图片
    image = preprocess(Image.open(os.path.join(pic_path, img))).unsqueeze(0)

    # 使用 ONNX 模型计算图像特征
    image_features = img_session.run(["unnorm_image_features"], {"image": image.cpu().numpy()})[0]  # 未归一化的图片特征
    image_features = torch.tensor(image_features)
    image_features /= image_features.norm(dim=-1, keepdim=True)  # 归一化后的Chinese-CLIP图像特征，用于下游任务
    print("image_features:", image_features.shape)
    # 将二维张量转换为一维
    image_features = image_features.squeeze()
    vector = image_features.tolist()

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
