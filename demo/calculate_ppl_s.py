import torch
from datasets import load_dataset
import torch.nn as nn
from tqdm import tqdm
from modelscope import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import gc
import os

# 设置更激进的内存管理
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def clean_memory():
    """清理GPU和CPU内存"""
    torch.cuda.empty_cache()
    gc.collect()

# 加载模型和分词器
model_name = '/root/autodl-tmp/Qwen2.5-0.5B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 使用fp16减少显存占用
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype='auto',
    device_map="auto",
)
model.eval()

# 加载数据集
dataset_path = "/root/autodl-fs/wikitext-test.arrow"
test_dataset = load_dataset("arrow", data_files=dataset_path)
test_text = "\n\n".join(test_dataset["train"]["text"])

# 动态获取模型的最大序列长度
max_length = model.config.max_position_embeddings
stride = 128  # 滑动窗口步长（可调整）

# ========== 关键修改：将数据集切分成20份 ==========
n_chunks = 20  # 切分成20份

chunk_size = len(test_text) // n_chunks
text_chunks = [test_text[i*chunk_size : (i+1)*chunk_size] for i in range(n_chunks)]

# 存储每份的PPL和有效token数
all_ppls = []
all_token_counts = []

# 遍历每个分片
for chunk_idx, chunk_text in enumerate(text_chunks):
    print(f"\nProcessing chunk {chunk_idx + 1}/{n_chunks}...")

    # 编码当前分片的文本
    input_ids = tokenizer.encode(chunk_text, return_tensors="pt")
    seq_len = input_ids.size(1)

    # 存储当前分片的NLL（负对数似然）和有效token数
    chunk_nlls = []
    chunk_token_counts = []

    # 滑动窗口计算
    for i in tqdm(range(0, seq_len, stride), desc=f"Chunk {chunk_idx + 1}"):
        start_idx = max(0, i + stride - max_length)
        end_idx = min(i + stride, seq_len)

        try:
            # 只将当前batch送到GPU
            batch = input_ids[:, start_idx:end_idx].to(model.device)

            with torch.inference_mode():  # 比torch.no_grad()更高效
                outputs = model(batch)
                logits = outputs.logits

                # 立即转移到CPU计算
                shift_logits = logits[:, :-1, :].contiguous().float().cpu()
                shift_labels = batch[:, 1:].contiguous().cpu()

                # 计算损失（逐token）
                loss_fct = nn.CrossEntropyLoss(reduction="none")
                losses = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )

                # 记录当前窗口的loss和有效token数
                valid_tokens = (end_idx - start_idx - 1)
                chunk_nlls.append(losses.sum().item())  # 累加loss
                chunk_token_counts.append(valid_tokens)

                # 清理GPU内存
                del batch, outputs, logits, shift_logits, shift_labels, losses
                clean_memory()

        except Exception as e:
            print(f"\nError at position {i}: {str(e)}")
            clean_memory()
            continue

            # 计算当前分片的PPL
    if chunk_nlls:
        total_nll = sum(chunk_nlls)
        total_tokens = sum(chunk_token_counts)
        chunk_ppl = torch.exp(torch.tensor(total_nll / total_tokens)).item()
        all_ppls.append(chunk_ppl)
        all_token_counts.append(total_tokens)
        print(f"Chunk {chunk_idx + 1} PPL: {chunk_ppl:.4f} (tokens: {total_tokens})")
    else:
        print(f"Chunk {chunk_idx + 1} failed to compute.")

    # ========== 汇总所有分片的PPL ==========
if all_ppls:
    # 加权平均（按token数）
    total_ppl = np.exp(
        sum(np.log(ppl) * tokens for ppl, tokens in zip(all_ppls, all_token_counts))
        / sum(all_token_counts)
    )
    print(f"\nFinal Perplexity (weighted average): {total_ppl:.4f}")
else:
    print("\nNo valid PPL calculated.")
# 鉴于单卡显存的不足被迫将文件拆成二十份每一份进行计算困惑度