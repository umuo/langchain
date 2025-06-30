#!/usr/bin/python3
# --*-- coding: utf-8 --*--
# @Author: gitsilence
# @Time: 2025/6/28 11:58
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- 1. 加载模型和分词器 ---
# 定义本地模型路径
model_name = "/Users/gitsilence/github/Qwen3-0.6B"

# 从指定路径加载分词器（Tokenizer）
# 分词器负责将文本转换为模型可以理解的数字ID（Tokens），以及将模型的输出ID转换回文本。
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 从指定路径加载预训练的因果语言模型（Causal LM）
# torch_dtype="auto" 让库自动选择最佳的数据类型（如 float16 或 bfloat16）以优化性能和内存。
# device_map="auto" 让库自动决定将模型的不同部分加载到哪个设备上（例如，GPU或CPU）。
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# --- 2. 准备模型输入 ---
# 定义用户输入的提示（Prompt）
prompt = "关于大语言模型，给我一个简短的介绍"
# 构建符合模型聊天格式的消息列表
messages = [
    {"role": "user", "content": prompt}
]

# 使用分词器的聊天模板来格式化输入
# 这是关键的“转换”步骤，它将结构化的消息列表转换为模型预期的单个字符串格式。
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,           # tokenize=False 表示此步骤只应用模板生成字符串，而不立即进行分词。
    add_generation_prompt=True, # add_generation_prompt=True 会在末尾添加一个特殊的提示，告诉模型它应该开始生成回复了。
    enable_thinking=False     # enable_thinking=False 关闭了模型的“思考”模式。如果为True，模型会先生成思考过程，再生成最终答案。
)
"""
<|im_start|>user
关于大语言模型，给我一个简短的介绍<|im_end|>
<|im_start|>assistant
<think>

</think>
"""
print(text)

# 对格式化后的文本进行分词，并转换为PyTorch张量（Tensors）
# return_tensors="pt" 表示返回PyTorch张量。
# .to(model.device) 将输入张量移动到模型所在的设备（例如GPU），以避免数据传输开销。
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# --- 3. 生成文本 ---
# 使用模型生成文本
# **model_inputs 将分词后的输入（input_ids, attention_mask等）解包并作为参数传入。
# max_new_tokens=32768 设置了模型最多可以生成的新词元（tokens）数量。
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)

# 从生成的结果中，切片掉原始输入的部分，只保留新生成的token ID。
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

# --- 4. 解析和解码输出 ---
# 尝试解析模型输出中的“思考”内容（如果存在）
# Qwen模型在“思考”模式下会使用特殊的token（ID为151668，代表 </think>）来分隔思考过程和最终答案。
try:
    # rindex 从列表末尾开始查找 151668 这个token的索引。
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    # 如果找不到该token，说明没有思考内容，索引设为0。
    index = 0

# 将思考部分的ID解码为文本
thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
# 将最终答案部分的ID解码为文本
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

# --- 5. 打印结果 ---
print("thinking content:", thinking_content)
print("content:", content)