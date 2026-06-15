# nanoLlama

A minimal, from-scratch implementation of a LLaMA-style language model in pure PyTorch.

Built to understand how modern LLMs actually work under the hood, covering every component from tokenization to training to generation.

## Architecture

This is a **decoder-only transformer** with the same design choices as LLaMA:

| Component | Choice | Notes |
|-----------|--------|-------|
| Positional encoding | **RoPE** | Relative positions, better length generalization than learned embeddings |
| Normalization | **RMSNorm** | Pre-norm placement; simpler and faster than LayerNorm |
| FFN activation | **SwiGLU** | Gated activation used in LLaMA, PaLM, and most modern LLMs |
| Tokenizer | **BPE** | Byte-pair encoding built from scratch, same algorithm as GPT-2 |
| Optimizer | **AdamW** | Decoupled weight decay, implemented from scratch |

## Setup

```sh
pip install uv
uv sync
source .venv/bin/activate
```

## Usage

### Train

```sh
python3 -m llmcore.train \
  --train_data data/train.bin \
  --val_data data/val.bin \
  --checkpoint_dir checkpoints \
  --vocab_size 10000 \
  --context_length 256 \
  --d_model 512 \
  --num_layers 6 \
  --num_heads 8 \
  --d_ff 1344 \
  --batch_size 32 \
  --total_steps 10000
```

### Generate

```sh
python3 -m llmcore.generate \
  --prompt "Once upon a time" \
  --temperature 0.8 \
  --top_k 50
```

`--temperature` controls randomness (lower = more conservative, higher = more creative). `--top_k` limits sampling to the top-k most likely tokens at each step.

### Tests

```sh
uv run pytest
```

## Data

```sh
mkdir -p data && cd data

# TinyStories — simple narrative text, good for quick training runs
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

# OpenWebText subsample — more diverse, web-scale text
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz
```
