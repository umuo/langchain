from idlelib.help_about import version

from opensearchpy import OpenSearch
import numpy as np
from PIL import Image
import torch

import onnxruntime
import cn_clip
from cn_clip.clip.utils import image_transform, _MODEL_INFO

from picture_search.pic_text_similar import text_features

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

# 载入ONNX文本侧模型
txt_sess_option = onnxruntime.SessionOptions()
txt_run_options = onnxruntime.RunOptions()
txt_run_options.log_severity_level = 2
txt_onnx_model_path = "/media/gitsilence/develop/PythonProject/Chinese-CLIP/cn_clip/deploy/vit-b-16.txt.fp32.onnx"
txt_session = onnxruntime.InferenceSession(
    txt_onnx_model_path,
    sess_options=txt_sess_option,
    providers=["CUDAExecutionProvider"]
)


model_arch = "ViT-B-16"
preprocess = image_transform(_MODEL_INFO[model_arch]['input_resolution'])


# 初始化 OpenSearch 客户端
client = OpenSearch(
    hosts=[{'host': 'localhost', 'port': 9200}],  # 替换为 OpenSearch 的地址和端口
    http_auth=('admin', 'gggMz123...'),  # 替换为你的用户名和密码
    use_ssl=True,  # 启用 HTTPS
    verify_certs=False,  # 忽略证书验证
    ssl_show_warn=False  # 禁用 SSL 警告
)
# 索引名称
index_name = "image-index"


def extract_image_features(image_path):
    """提取图片特征向量"""
    # 预处理图片
    image = preprocess(Image.open(image_path)).unsqueeze(0)

    # 使用 ONNX 模型计算图像特征
    image_features = img_session.run(["unnorm_image_features"], {"image": image.cpu().numpy()})[0]  # 未归一化的图片特征
    image_features = torch.tensor(image_features)
    image_features /= image_features.norm(dim=-1, keepdim=True)  # 归一化后的Chinese-CLIP图像特征，用于下游任务
    print("image_features:", image_features.shape)
    # 将二维张量转换为一维
    image_features = image_features.squeeze()
    vector = image_features.tolist()
    return vector

def search_by_image(image_path, top_k=5):
    """以图搜图"""
    vector = extract_image_features(image_path)
    # print(vector)
    query = {
        "query": {
            "knn": {
                "image_vector": {
                    "vector": vector,
                    "k": top_k
                }
            }
        }
    }
    response = client.search(index=index_name, body=query)
    return response['hits']['hits']


def search_by_text_vertor(query_text, top_k=5):
    """
    获取搜索关键字的向量
    :param query_text:
    :param top_k:
    :return:
    """
    text = cn_clip.clip.tokenize(query_text.split(","), context_length=52)
    text_features = []
    for i in range(len(text)):
        one_text = np.expand_dims(text[i].cpu().numpy(), axis=0)
        text_feature = txt_session.run(["unnorm_text_features"], {"text": one_text})[0]  # 未归一化的文本特征
        text_feature = torch.tensor(text_feature)
        text_features.append(text_feature)
    text_features = torch.squeeze(torch.stack(text_features), dim=1)  # 多个特征向量stack到一起
    text_features = text_features / text_features.norm(dim=1, keepdim=True)  # 归一化后的Chinese-CLIP文本特征，用于下游任务
    # 将二维张量转换为一维
    image_features = text_features.squeeze()
    vector = image_features.tolist()
    query = {
        "query": {
            "knn": {
                "image_vector": {
                    "vector": vector,
                    "k": top_k
                }
            }
        }
    }
    response = client.search(index=index_name, body=query)
    return response['hits']['hits']
    pass


def search_by_text(query_text, top_k=5):
    """以文搜图"""
    query = {
        "query": {
            "match": {
                "ocr_text": {
                    "query": query_text,
                    "fuzziness": "AUTO"  # 支持模糊匹配
                }
            }
        },
        "size": top_k
    }

    response = client.search(index=index_name, body=query)
    return response['hits']['hits']


# 示例测试
if __name__ == "__main__":
    # 以图搜图
    image_path = "../ocr/img.png"  # 替换为你的图片路径
    print("以图搜图结果：")
    image_results = search_by_image(image_path)
    for hit in image_results:
        print(f"Score: {hit['_score']}, Image Path: {hit['_source']['image_path']}")

    # 以文搜图
    query_text = "计算机"  # 替换为你的搜索关键字
    print("\n以文搜图结果：")
    text_results = search_by_text(query_text)
    for hit in text_results:
        print(
            f"Score: {hit['_score']}, OCR Text: {hit['_source']['ocr_text']}, Image Path: {hit['_source']['image_path']}")

    query_text = "手机"  # 替换为你的搜索关键字
    print("\n以特征描述搜图结果：")
    text_results = search_by_text_vertor(query_text)
    for hit in text_results:
        print(
            f"Score: {hit['_score']}, OCR Text: {hit['_source']['ocr_text']}, Image Path: {hit['_source']['image_path']}")
