# train_safety_discriminator.py
"""
使用 PKU-SafeRLHF 数据集训练多分类安全鉴别器。

多分类标签 (5 类):
  0 - Safe              安全内容
  1 - Offensive         攻击性/仇恨言论
  2 - Physical Harm     身体/心理伤害
  3 - Fraud/Financial   欺诈/经济伤害
  4 - Privacy/Legal     隐私/违法

训练完成后模型保存至 ./safety_discriminator_multiclass/
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix
from config import Config

# ─────────────────────────────────────────────
# 超参数
# ─────────────────────────────────────────────
MODEL_CHECKPOINT = "distilbert-base-uncased"
OUTPUT_DIR       = "./safety_discriminator_multiclass"
MAX_LEN          = 256
BATCH_SIZE       = 16
NUM_EPOCHS       = 3
LR               = 2e-5
WARMUP_RATIO     = 0.1
MAX_TRAIN_SAMPLES = 20_000   # 设 None 则使用全部
MAX_EVAL_SAMPLES  = 2_000

NUM_LABELS = 5
LABEL_NAMES = [
    "Safe",           # 0
    "Offensive",      # 1
    "Physical Harm",  # 2
    "Fraud/Financial",# 3
    "Privacy/Legal",  # 4
]

# ─────────────────────────────────────────────
# 标签映射
# ─────────────────────────────────────────────
def harm_to_label(is_safe: bool, harm_types: dict) -> int:
    """
    将 PKU-SafeRLHF 的安全标记和伤害类别转换为多分类标签。

    harm_types 是一个字典，键是伤害类型名称，值是布尔值。
    PKU-SafeRLHF 的伤害类别字段（根据数据集文档）：
      - 'offensiveness'  / 'hate_speech'
      - 'physical_harm'  / 'physical_safety'
      - 'psychological_harm'
      - 'financial_harm' / 'economic_harm' / 'fraud'
      - 'privacy_violation' / 'privacy' / 'legal'
    """
    if is_safe:
        return 0  # Safe

    if harm_types is None:
        return 1  # 默认归为 Offensive

    def _check(keys):
        return any(harm_types.get(k, False) for k in keys)

    if _check(["offensiveness", "hate_speech", "discrimination"]):
        return 1  # Offensive
    if _check(["physical_harm", "physical_safety", "psychological_harm",
               "violence", "self_harm"]):
        return 2  # Physical Harm
    if _check(["financial_harm", "economic_harm", "fraud", "deception"]):
        return 3  # Fraud/Financial
    if _check(["privacy_violation", "privacy", "legal", "copyright"]):
        return 4  # Privacy/Legal

    return 1  # 兜底归为 Offensive


def extract_samples_from_dataset(split_data, max_samples=None):
    """
    从 PKU-SafeRLHF 数据集提取 (text, label) 对。
    每条记录包含两个 response，各自独立提取。
    """
    samples = []
    for row in split_data:
        prompt = row.get("prompt", "")

        for idx in range(2):
            response_key = f"response_{idx}"
            safe_key     = f"is_response_{idx}_safe"
            harm_key     = f"response_{idx}_harm_category"

            response  = row.get(response_key, "")
            is_safe   = row.get(safe_key, True)
            harm_cats = row.get(harm_key, None)

            if not response:
                continue

            # 拼接 prompt + response，让模型同时看到上下文
            text  = f"{prompt} [SEP] {response}"
            label = harm_to_label(bool(is_safe), harm_cats)
            samples.append((text, label))

            if max_samples and len(samples) >= max_samples:
                return samples

    return samples


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class SafetyDataset(Dataset):
    def __init__(self, samples, tokenizer, max_len):
        self.samples   = samples
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(label, dtype=torch.long),
        }


# ─────────────────────────────────────────────
# 训练
# ─────────────────────────────────────────────
def train():
    device = torch.device(Config.DEVICE)
    print(f"[Train] Device: {device}")

    # 1. 加载数据集
    print("[Train] Loading PKU-SafeRLHF dataset...")
    raw = load_dataset("PKU-Alignment/PKU-SafeRLHF", trust_remote_code=True)

    train_split = raw["train"]
    eval_split  = raw.get("test", raw.get("validation", raw["train"].select(range(min(5000, len(raw["train"]))))))

    print(f"[Train] Raw train size: {len(train_split)}, eval size: {len(eval_split)}")

    train_samples = extract_samples_from_dataset(train_split, MAX_TRAIN_SAMPLES)
    eval_samples  = extract_samples_from_dataset(eval_split,  MAX_EVAL_SAMPLES)

    # 打印标签分布
    from collections import Counter
    train_dist = Counter(s[1] for s in train_samples)
    print(f"[Train] Train label distribution: { {LABEL_NAMES[k]: v for k, v in sorted(train_dist.items())} }")

    # 2. Tokenizer & Model
    print(f"[Train] Loading tokenizer & model: {MODEL_CHECKPOINT}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=NUM_LABELS,
    ).to(device)

    # 3. DataLoaders
    train_ds = SafetyDataset(train_samples, tokenizer, MAX_LEN)
    eval_ds  = SafetyDataset(eval_samples,  tokenizer, MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    eval_loader  = DataLoader(eval_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 4. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps   = len(train_loader) * NUM_EPOCHS
    warmup_steps  = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # 5. 训练循环
    best_eval_acc = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        # ── Train ──────────────────────────────
        model.train()
        total_loss, total_correct, total_count = 0.0, 0, 0

        for step, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss    = outputs.loss
            logits  = outputs.logits

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            preds = logits.argmax(dim=-1)
            total_loss    += loss.item() * labels.size(0)
            total_correct += (preds == labels).sum().item()
            total_count   += labels.size(0)

            if (step + 1) % 100 == 0:
                running_acc = total_correct / total_count
                print(f"  Epoch {epoch} | Step {step+1}/{len(train_loader)} "
                      f"| Loss={total_loss/total_count:.4f} | Acc={running_acc:.4f}")

        train_acc  = total_correct / total_count
        train_loss = total_loss    / total_count
        print(f"[Epoch {epoch}] Train Loss={train_loss:.4f}  Acc={train_acc:.4f}")

        # ── Eval ───────────────────────────────
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in eval_loader:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds   = outputs.logits.argmax(dim=-1)

                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        eval_acc = np.mean(np.array(all_preds) == np.array(all_labels))
        print(f"[Epoch {epoch}] Eval  Acc={eval_acc:.4f}")
        print(classification_report(all_labels, all_preds, target_names=LABEL_NAMES, zero_division=0))

        # 保存最佳模型
        if eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            print(f"  ✔ Saved best model (acc={best_eval_acc:.4f}) → {OUTPUT_DIR}")

    print(f"\n[Train] Done. Best eval acc = {best_eval_acc:.4f}")
    print(f"[Train] Model saved to: {OUTPUT_DIR}")


# ─────────────────────────────────────────────
if __name__ == "__main__":
    train()
