# --coding: utf-8--
import tiktoken

"""
tiktoken 是open ai 的一个库，用于计算文本的token数量
"""


def count_tokens(text: str, model: str = "gpt-3.5-turbo-0301") -> int:
    """
    计算文本的token数量
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print(f"Warning: Model '{model}' not directly supported by tiktoken. Using 'cl100k_base' as a fallback.")
        encoding = tiktoken.get_encoding("cl100k_base")
    # print(encoding)
    tokens = encoding.encode(text)
    decoded_tokens = [encoding.decode_single_token_bytes(token) for token in tokens]
    print(f"Total tokens: {len(tokens)}")
    print("Token ID | Token String")
    print("------- | --------")
    for token_id, decoded_token in zip(tokens, decoded_tokens):
        #  decode_single_token_bytes() 返回的是 bytes 类型，需要转换为字符串
        print(f"{token_id:8d} | {decoded_token.decode('utf-8', errors='replace')}")
    num_tokens = len(encoding.encode(text))
    return num_tokens


if __name__ == '__main__':
    count = count_tokens("我是一只小小小鸟", model="gpt-4o")
    print(count)
