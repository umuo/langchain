
from paddleocr import PaddleOCR

# Paddleocr目前支持中英文、英文、法语、德语、韩语、日语，可以通过修改lang参数进行切换
# 参数依次为`ch`, `en`, `french`, `german`, `korean`, `japan`。
ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # need to run only once to download and load model into memory
# 选择你要识别的图片路径
img_path = 'img.png'
result = ocr.ocr(img_path, cls=True)

str_list = []
# 打印识别结果
for line in result[0]:  # results[0] 包含文字区域和识别结果
    str_list.append(line[1][0])
    # print("文字内容:", line[1][0])  # line[1][0] 是识别出的文字
    # print("置信度:", line[1][1])  # line[1][1] 是置信度
print(",".join(str_list))
