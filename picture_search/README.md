
# 实现图片检索

## 安装 OpenSearch

启动前的配置：
```bash
sudo sysctl -w vm.max_map_count=262144
```

启动命令
```bash
OPENSEARCH_INITIAL_ADMIN_PASSWORD=gggMz888. docker compose up -d
```
启动成功后，可以访问 http://localhost:5601 查看是否启动成功
账户名：admin
密码：gggMz888.

## 创建索引

```bash
curl -X PUT "http://localhost:9200/image-index" -H 'Content-Type: application/json' -d'
{
  "settings": {
    "number_of_shards": 1,
    "number_of_replicas": 0
  },
  "mappings": {
    "properties": {
      "image_vector": {
        "type": "knn_vector",   # 用于 kNN 检索
        "dimension": 512        # 特征向量的维度，比如 CLIP 输出的是 512 维向量
      },
      "ocr_text": {
        "type": "text"          # 用于存储 OCR 的文本
      },
      "image_path": {
        "type": "keyword"       # 用于存储图片路径
      }
    }
  }
}'
```
或者
```bash
PUT image-index
{
  "settings": {
    "number_of_shards": 1,
    "number_of_replicas": 0,
    "index": {
      "knn": true
    }
  },
  "mappings": {
    "properties": {
      "image_vector": {
        "type": "knn_vector", 
        "dimension": 512       
      },
      "ocr_text": {
        "type": "text"         
      },
      "image_path": {
        "type": "keyword"      
      }
    }
  }
}
```
## 图片批量采集
来源：https://www.pexels.com/zh-cn/search/%E6%97%A5%E5%B8%B8%E7%89%A9%E5%93%81/

手工下载几十张

搭建一个临时的文件服务器，可以通过URL访问到图片

## 图片信息入库（图片特征提取、图片OCR）

### 图片批量入库
参考代码：picture_deal.py

## 检索测试
### 图片检索

### 图片内容检索

### 图片文字转向量检索

## 总结
总共测试两个模型向量化，检索的前两个结果还是很明显的，后面的就不太匹配了。
还需要多找些样本测试。

https://github.com/OFA-Sys/Chinese-CLIP
