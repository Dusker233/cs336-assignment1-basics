import os
import re
import regex

from collections import Counter

def make_bpe_tokenizer(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Constructs a BPE tokenizer from the given training data.
    
    Args:
        input_path: Path to the training data file.
        vocab_size: The target vocabulary size.
        special_tokens: List of special tokens to include in the vocabulary.
    Returns:
        A tuple containing:
            - vocab: A dictionary mapping token IDs to byte strings.
            - merges: A list of tuples representing the merge operations.
    """
    # 1. init vocab size=256
    size = 256
    vocab = {i: bytes([i]) for i in range(size)}
    num_iterations = vocab_size - size - len(special_tokens)
    # 2. read training data
    with open(input_path, "r", encoding="utf-8") as f:
        train_data = f.read()
    # 3. pre-tokenize the training data by special tokens
    if special_tokens:
        special_regex = "|".join(re.escape(token) for token in special_tokens)
        train_parts = re.split(f"({special_regex})", train_data)
        train_parts = [part for part in train_parts if part not in special_tokens]
    else:
        train_parts = [train_data]
    # 4. pre-tokenize train_parts with GPT-2 tokenizer
    PAT = regex.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    word_freq = Counter()
    for part in train_parts:
        words = PAT.findall(part)
        # count byte for each word
        for word in words:
            word_freq[tuple(word.encode("utf-8"))] += 1
    merges = []
    for _ in range(num_iterations):
        # count byte pair for each word
        pair_freq = Counter()
        for word, freq in word_freq.items():
            if len(word) < 2:
                continue
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_freq[pair] += freq
        if not pair_freq:
            break
        # find most common pair
        best = max(pair_freq, key=lambda p: (pair_freq[p], vocab[p[0]], vocab[p[1]]))
        left, right = vocab[best[0]], vocab[best[1]]
        merges.append((left, right))
        vocab[size] = left + right
        # merge pair for each word
        new_word_freq = Counter()
        for word, freq in word_freq.items():
            if len(word) < 2:
                new_word_freq[word] = freq
                continue
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == best[0] and word[i + 1] == best[1]:
                    new_word.append(size)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_freq[tuple(new_word)] += freq
        word_freq = new_word_freq
        size += 1
    for token in special_tokens:
        vocab[size] = token.encode("utf-8")
        size += 1
    return vocab, merges