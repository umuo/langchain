from opensearchpy import OpenSearch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
# 初始化 OpenSearch 客户端
client = OpenSearch(
    hosts=[{'host': 'localhost', 'port': 9200}],  # 替换为 OpenSearch 的地址和端口
    http_auth=('admin', 'gggMz88.'),                # 替换为你的用户名和密码
    use_ssl=True,                                # 启用 HTTPS
    verify_certs=False,                          # 忽略证书验证
    ssl_show_warn=False                          # 禁用 SSL 警告
)

# 初始化 CLIP 模型
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 索引名称
index_name = "image-index"

def extract_image_features(image_path):
    """提取图片特征向量"""
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt", do_center_crop=True)
    with torch.no_grad():
        vector = clip_model.get_image_features(**inputs).squeeze().numpy()
    vector /= np.linalg.norm(vector)  # 归一化
    return vector.astype(float).tolist()

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
    inputs = clip_processor(text=query_text, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = clip_model.get_text_features(**inputs)
    query = {
        "query": {
            "knn": {
                "image_vector": {
                    "vector": outputs[0].tolist(),
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
        print(f"Score: {hit['_score']}, OCR Text: {hit['_source']['ocr_text']}, Image Path: {hit['_source']['image_path']}")

    query_text = "美女"  # 替换为你的搜索关键字
    print("\n以特征描述搜图结果：")
    text_results = search_by_text_vertor(query_text)
    for hit in text_results:
        print(f"Score: {hit['_score']}, OCR Text: {hit['_source']['ocr_text']}, Image Path: {hit['_source']['image_path']}")

