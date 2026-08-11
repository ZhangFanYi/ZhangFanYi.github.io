import os
import json
import glob

import torch
from torch.utils.data import Dataset
from qwen_vl_utils import process_vision_info


def find_assistant_spans(token_ids, tokenizer):
    """
    定位 Qwen3.5 VLM 中所有 assistant 回复的 token 范围。
    通过 tokenizer 动态解析 token ID，兼容 Qwen3.5 全系列。

    返回 [(start, end), ...] 列表，start 为内容起始（不含 header），end 包含 <|im_end|>。
    """
    im_start_ids = tokenizer.encode("<|im_start|>", add_special_tokens=False)
    im_end_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    assistant_tokens = tokenizer.encode("assistant\n", add_special_tokens=False)
    start_seq = im_start_ids + assistant_tokens

    im_end_id = im_end_ids[0]  # <|im_end|> 通常是单个 token
    len_start = len(start_seq)

    spans = []
    i = 0
    while i <= len(token_ids) - len_start:
        if token_ids[i:i + len_start] == start_seq:
            content_start = i + len_start
            # 查找对应的 <|im_end|>
            content_end = len(token_ids)
            for j in range(content_start, len(token_ids)):
                if token_ids[j] == im_end_id:
                    content_end = j + 1  # 包含 <|im_end|>
                    break
            spans.append((content_start, content_end))
            i = content_end
        else:
            i += 1
    return spans


class SimpleDataset(Dataset):
    """从目录中加载所有 .jsonl 文件的最小 Dataset。"""

    def __init__(self, data_path):
        self.data = []
        if os.path.isfile(data_path):
            jsonl_files = [data_path]
        else:
            jsonl_files = sorted(glob.glob(os.path.join(data_path, "*.jsonl")))
        if not jsonl_files:
            raise FileNotFoundError(f"No .jsonl files found in {data_path}")
        for filepath in jsonl_files:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.data.append(json.loads(line))
        print(f"Loaded {len(self.data)} samples from {len(jsonl_files)} files")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def make_collate_fn(processor, max_seq_length):
    """
    创建 Qwen3.5 VLM 多模态 collate 函数。
    处理文本+图片输入，仅对 assistant 回复部分计算 loss。
    """
    tokenizer = processor.tokenizer

    def collate_fn(batch):
        messages = [item["messages"] for item in batch]

        # 文本处理：应用 chat template
        texts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
            for msg in messages
        ]

        # 视觉信息处理
        image_inputs, video_inputs = process_vision_info(messages)

        # 整合输入（tokenization + 图像预处理）
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
        )

        # 创建 labels：-100 mask 非 assistant 部分
        input_ids = inputs["input_ids"]
        labels_list = []
        for ids in input_ids.tolist():
            label = [-100] * len(ids)
            for start, end in find_assistant_spans(ids, tokenizer):
                label[start:end] = ids[start:end]
            labels_list.append(label)

        inputs["labels"] = torch.tensor(labels_list, dtype=torch.long)
        return inputs

    return collate_fn
