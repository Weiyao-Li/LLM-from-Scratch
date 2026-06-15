import time
import tracemalloc
import pickle
from cs336_basics.bpe import train_bpe

INPUT_PATH = "data/TinyStoriesV2-GPT4-train.txt"
VOCAB_SIZE = 10_000
SPECIAL_TOKENS = ["<|endoftext|>"]

'''
run command:
cd /Users/nolan.li/Desktop/336/assignment1-basics
python3 cs336_basics/train_bpe_tinystories.py
'''
if __name__ == '__main__':
    tracemalloc.start()
    t0 = time.time()

    vocab, merges = train_bpe(
        input_path=INPUT_PATH,
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
    )

    t1 = time.time()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"训练耗时: {t1 - t0:.2f}s")
    print(f"峰值内存: {peak / 1024**2:.1f} MB")

    longest_id = max(vocab, key=lambda k: len(vocab[k]))
    print(f"最长 token id={longest_id}: {vocab[longest_id]}")
    print(f"最长 token 长度: {len(vocab[longest_id])} bytes")

    with open("data/tinystories_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open("data/tinystories_merges.pkl", "wb") as f:
        pickle.dump(merges, f)

    print("vocab 和 merges 已保存到 data/")