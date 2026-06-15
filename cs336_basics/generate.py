import argparse
import pickle
import torch
import torch.nn.functional as F

from cs336_basics.module import TransformerLM, load_checkpoint
from cs336_basics.tokenizer import Tokenizer


def sample_next_token(logits: torch.Tensor, temperature: float, top_k: int) -> int:
    # 用 temperature 缩放 logits：
    # temperature < 1 → 分布更尖锐（更保守）；temperature > 1 → 分布更平坦（更随机）
    logits = logits / temperature

    # Top-k 截断：只保留概率最高的 k 个 token，其余设为 -inf（softmax 后概率趋近 0）
    # 这样可以避免模型采样到极低概率的离奇词
    if top_k > 0:
        topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        threshold = topk_vals[..., -1]  # 第 k 大的值作为截断阈值
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    probs = F.softmax(logits, dim=-1)
    # 按概率分布随机采样一个 token id
    return torch.multinomial(probs, num_samples=1).item()


def generate(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: torch.device,
    eos_id: int | None,
) -> str:
    model.eval()

    # 将 prompt 文本编码为 token id 列表
    ids = tokenizer.encode(prompt)
    # 从模型获取最大上下文窗口长度
    context_length = model.context_length
    # shape: (1, seq_len)，batch size 为 1
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 截取最近 context_length 个 token，防止超出模型位置编码范围
            input_window = input_ids[:, -context_length:]

            # 前向传播，得到每个位置的 logits，shape: (1, seq_len, vocab_size)
            logits = model(input_window)

            # 只取最后一个位置的 logits 用于预测下一个 token，shape: (vocab_size,)
            next_logits = logits[0, -1, :]

            # 通过 temperature + top-k 采样得到下一个 token id
            next_id = sample_next_token(next_logits, temperature, top_k)

            # 遇到 eos token 则停止生成
            if eos_id is not None and next_id == eos_id:
                break

            # 将新 token 追加到序列末尾，进入下一轮自回归生成
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1
            )

    # 去掉原始 prompt 部分，只返回新生成的内容
    generated_ids = input_ids[0, len(ids):].tolist()
    return tokenizer.decode(generated_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ckpt_0010000.pt")
    parser.add_argument("--vocab", type=str, default="data/tinystories_vocab.pkl")
    parser.add_argument("--merges", type=str, default="data/tinystories_merges.pkl")
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # 加载 BPE tokenizer 的词表和合并规则
    with open(args.vocab, "rb") as f:
        vocab = pickle.load(f)
    with open(args.merges, "rb") as f:
        merges = pickle.load(f)
    tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=["<|endoftext|>"])
    # 获取 eos token 的 id，生成时遇到它就停止
    eos_id = tokenizer.vocab_to_id.get("<|endoftext|>".encode("utf-8"))

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
    )
    # 从 checkpoint 恢复模型权重（推理阶段不需要 optimizer）
    load_checkpoint(args.checkpoint, model, optimizer=None)
    model.to(device)

    print(f"Prompt: {args.prompt}")
    print("-" * 40)
    output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
        eos_id=eos_id,
    )
    # 拼接 prompt 和生成内容一起打印
    print(args.prompt + output)


if __name__ == "__main__":
    main()
