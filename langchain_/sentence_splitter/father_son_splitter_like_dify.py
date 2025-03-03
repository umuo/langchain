from langchain_community.document_loaders import UnstructuredMarkdownLoader, Docx2txtLoader
import tiktoken
from typing import Iterable, List, Optional

"""
dify的逻辑：
1、首先将文档根据 '\n\n' 进行split
2、然后判断当前tokens是否大于chun_size，如果大于继续后面的操作，否则就pass，使用当前chunk
3、["\n\n", "。", ". ", " ", ""]，首先判断第一个字符串，在原文中是否存在，如果存在就进行split，否则继续判断第二个字符串，以此类推
4、split之后，分别计算tokens，如果小于chunk_size，则先判断当前chunk的token+下一个chunk的token是否大于chunk_size，如果大于忽略，否则将进行合并chunk，
5、让单个chunk块的token 为小于等于chunk_size的最大值
6、以上为 父文档的splitter
7、遍历每个父文档，分别再进行splitter
"""


def computer_token(text: str, model: str = "gpt2") -> int:
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
    # print(f"Total tokens: {len(tokens)}")
    return len(tokens)


def split_and_merge_by_chunk_size(texts: list[str], chunk_size: int, separators: list[str] = ["。", ". ", " ", ""]) -> \
        list[str]:
    """
    将文本列表按照chunk_size进行分割和合并
    Args:
        texts: 待处理的文本列表
        chunk_size: 每个chunk的最大token数量
        separators: 分隔符列表，按优先级排序
    Returns:
        处理后的文本列表，每个元素的token数都小于等于chunk_size
    """
    result = []
    temp_chunk = []
    current_tokens = 0

    def merge_chunks(chunks: list[str], separator: str) -> None:
        nonlocal temp_chunk, current_tokens, result
        if not chunks:
            return

        # 将当前累积的chunk添加到结果中
        if temp_chunk:
            result.append(separator.join(temp_chunk))
            temp_chunk = []
            current_tokens = 0

        # 处理新的chunks
        for chunk in chunks:
            chunk_tokens = computer_token(chunk)
            if chunk_tokens > chunk_size:
                # 如果单个chunk超过限制，需要继续分割
                for sep in separators:
                    if sep in chunk:
                        sub_chunks = chunk.split(sep)
                        merge_chunks(sub_chunks, sep)
                        break
            else:
                # 尝试将chunk添加到当前temp_chunk
                potential_chunk = temp_chunk + [chunk] if temp_chunk else [chunk]
                potential_text = separator.join(potential_chunk)
                potential_tokens = computer_token(potential_text)

                if potential_tokens <= chunk_size:
                    temp_chunk = potential_chunk
                    current_tokens = potential_tokens
                else:
                    # 如果添加后超过限制，先保存当前temp_chunk
                    if temp_chunk:
                        result.append(separator.join(temp_chunk))
                    temp_chunk = [chunk]
                    current_tokens = chunk_tokens

    # 处理主文本列表
    merge_chunks(texts, separators[0])

    # 处理最后剩余的temp_chunk
    if temp_chunk:
        result.append(separators[0].join(temp_chunk))

    return result


def _length_function(splits: list[str]) -> list[int]:
    """Calculate the token length of each split."""
    return [computer_token(s) for s in splits]


def _join_docs(docs: list[str], separator: str) -> Optional[str]:
    text = separator.join(docs)
    text = text.strip()
    if text == "":
        return None
    else:
        return text


global _chunk_size, _fixed_separator
_fixed_separator = "\n"
_chunk_size = 500
_chunk_overlap = 0
_separators = ["\n\n", "。", ". ", " ", ""]


def _merge_splits(splits: Iterable[str], separator: str, lengths: list[int]) -> list[str]:
    # We now want to combine these smaller pieces into medium size
    # chunks to send to the LLM.
    separator_len = _length_function([separator])[0]

    docs = []
    current_doc: list[str] = []
    total = 0
    index = 0
    for d in splits:
        _len = lengths[index]
        if total + _len + (separator_len if len(current_doc) > 0 else 0) > _chunk_size:
            if total > _chunk_size:
                logger.warning(
                    f"Created a chunk of size {total}, which is longer than the specified {_chunk_size}"
                )
            if len(current_doc) > 0:
                doc = _join_docs(current_doc, separator)
                if doc is not None:
                    docs.append(doc)
                # Keep on popping if:
                # - we have a larger chunk than in the chunk overlap
                # - or if we still have any chunks and the length is long
                while total > _chunk_overlap or (
                        total + _len + (separator_len if len(current_doc) > 0 else 0) > _chunk_size and total > 0
                ):
                    total -= _length_function([current_doc[0]])[0] + (
                        separator_len if len(current_doc) > 1 else 0
                    )
                    current_doc = current_doc[1:]
        current_doc.append(d)
        total += _len + (separator_len if len(current_doc) > 1 else 0)
        index += 1
    doc = _join_docs(current_doc, separator)
    if doc is not None:
        docs.append(doc)
    return docs


def recursive_split_text(text: str) -> list[str]:
    """Split incoming text and return chunks."""
    final_chunks = []
    # Get appropriate separator to use
    separator = _separators[-1]
    for _s in _separators:
        if _s == "":
            separator = _s
            break
        if _s in text:
            separator = _s
            break
    # Now that we have the separator, split the text
    if separator:
        splits = text.split(separator)
    else:
        splits = list(text)
    # Now go merging things, recursively splitting longer texts.
    _good_splits = []
    _good_splits_lengths = []  # cache the lengths of the splits
    s_lens = _length_function(splits)
    for s, s_len in zip(splits, s_lens):
        if s_len < _chunk_size:
            _good_splits.append(s)
            _good_splits_lengths.append(s_len)
        else:
            if _good_splits:
                merged_text = _merge_splits(_good_splits, separator, _good_splits_lengths)
                final_chunks.extend(merged_text)
                _good_splits = []
                _good_splits_lengths = []
            other_info = recursive_split_text(s)
            final_chunks.extend(other_info)
    if _good_splits:
        merged_text = _merge_splits(_good_splits, separator, _good_splits_lengths)
        final_chunks.extend(merged_text)
    return final_chunks


def split_text(text: str) -> list[str]:
    """Split incoming text and return chunks."""
    if _fixed_separator:
        chunks = text.split(_fixed_separator)
    else:
        chunks = [text]

    final_chunks = []
    chunks_lengths = _length_function(chunks)
    for chunk, chunk_length in zip(chunks, chunks_lengths):
        if chunk_length > _chunk_size:
            final_chunks.extend(recursive_split_text(chunk))
        else:
            final_chunks.append(chunk)
    return final_chunks


# loader = Docx2txtLoader(
#     r"/Users/gitsilence/Documents/弘扬传统文化.docx"
# )

loader = UnstructuredMarkdownLoader(
    r"E:\DeskTop\deep_research_test_data.md"
)

documents = loader.load()
# print(documents)
print(f"加载了 {len(documents)} 个文档")
content = documents[0].page_content
content = content.replace("\n\n", "\n")

# print(content)
# 首先根据 \n\n 进行split

# print(content)
_fixed_separator = "\n\n"
parent_docs = split_text(content)
# for doc in parent_docs:
#     print(doc)
#     print("----" * 20)
print("****" * 20)

son_docs = []

_chunk_size = 200
_fixed_separator = "\n"
for doc in parent_docs:
    print(f"\t父文档：{doc}")
    print("----" * 20)
    child_docs = split_text(doc)
    son_docs.extend(child_docs)
    for child_doc in child_docs:
        print(f"\t子文档：{child_doc}")
        print("****" * 20)

# for doc in son_docs:
#     print(doc)
#     print("****" * 20)
