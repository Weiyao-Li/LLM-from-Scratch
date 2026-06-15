import pickle
import numpy as np
from cs336_basics.tokenizer import Tokenizer

VOCAB_PATH  = "data/tinystories_vocab.pkl"
MERGES_PATH = "data/tinystories_merges.pkl"
TRAIN_TXT   = "data/TinyStoriesV2-GPT4-train.txt"
VAL_TXT     = "data/TinyStoriesV2-GPT4-valid.txt"
TRAIN_BIN   = "data/train.bin"
VAL_BIN     = "data/val.bin"
SPECIAL_TOKENS = ["<|endoftext|>"]

def tokenize_file_to_bin(txt_path, bin_path, tokenizer):
    import os
    import sys
    total_bytes = os.path.getsize(txt_path)
    ids = []
    read_bytes = 0
    last_pct = -1
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            read_bytes += len(line.encode("utf-8"))
            for token_id in tokenizer.encode_iterable([line]):
                ids.append(token_id)
            pct = int(read_bytes / total_bytes * 100)
            if pct != last_pct:
                last_pct = pct
                sys.stdout.write(f"\r{txt_path}: {pct}%  ({len(ids):,} tokens)")
                sys.stdout.flush()
    print()
    arr = np.array(ids, dtype=np.uint16)
    arr.tofile(bin_path)
    print(f"{txt_path} -> {bin_path}  ({len(arr):,} tokens)")

if __name__ == "__main__":
    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)
    with open(MERGES_PATH, "rb") as f:
        merges = pickle.load(f)

    tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=SPECIAL_TOKENS)

    tokenize_file_to_bin(TRAIN_TXT, TRAIN_BIN, tokenizer)
    tokenize_file_to_bin(VAL_TXT,   VAL_BIN,   tokenizer)

    print("数据准备完成！")
