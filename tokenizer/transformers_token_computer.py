from transformers import GPT2Tokenizer as TransformerGPT2Tokenizer
from os.path import abspath, dirname, join

base_path = abspath(__file__)
gpt2_tokenizer_path = join(dirname(base_path), "gpt2")
_tokenizer = TransformerGPT2Tokenizer.from_pretrained(gpt2_tokenizer_path)
tokens = _tokenizer.encode("hello world")

print(len(tokens))