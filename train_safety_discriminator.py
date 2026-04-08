import os
from collections import Counter

import numpy as np
import torch
from config import Config
from dataset_utils import load_local_pku_dataset
from hf_utils import resolve_model_path
from risk_taxonomy import NUM_RISK_LABELS, RISK_LABEL_NAMES, harm_types_to_multihot
from utils import set_seed
from wandb_logger import WandbLogger
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


MODEL_CHECKPOINT = "distilbert-base-uncased"
OUTPUT_DIR = "./safety_discriminator_multilabel"
POS_WEIGHT_FILE = "pos_weight.npy"
MAX_LEN = 256
BATCH_SIZE = 16
NUM_EPOCHS = 3
LR = 2e-5
WARMUP_RATIO = 0.1
MAX_TRAIN_SAMPLES = 20_000
MAX_EVAL_SAMPLES = 2_000


class SafetyDataset(Dataset):
    def __init__(self, samples, tokenizer, max_len):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, labels = self.samples[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(labels, dtype=torch.float32),
        }


def row_to_multihot(is_safe: bool, harm_types) -> np.ndarray:
    if is_safe:
        return np.zeros(NUM_RISK_LABELS, dtype=np.float32)

    labels = harm_types_to_multihot(harm_types)
    if labels.sum() == 0:
        labels[1] = 1.0
    return labels


def extract_samples_from_dataset(split_data, max_samples=None):
    samples = []
    for row in split_data:
        prompt = row.get("prompt", "")

        for idx in range(2):
            response = row.get(f"response_{idx}", "")
            is_safe = bool(row.get(f"is_response_{idx}_safe", True))
            harm_cats = row.get(f"response_{idx}_harm_category", None)

            if not response:
                continue

            text = f"{prompt} [SEP] {response}"
            labels = row_to_multihot(is_safe, harm_cats)
            samples.append((text, labels))

            if max_samples and len(samples) >= max_samples:
                return samples

    return samples


def multilabel_metrics(all_labels, all_probs):
    probs = np.asarray(all_probs)
    labels = np.asarray(all_labels)
    preds = (probs >= 0.5).astype(int)
    exact_match = np.all(preds == labels, axis=1).mean()
    micro_acc = (preds == labels).mean()
    return exact_match, micro_acc, preds


def compute_pos_weight(samples):
    label_matrix = np.stack([labels for _, labels in samples], axis=0)
    pos_counts = label_matrix.sum(axis=0)
    neg_counts = label_matrix.shape[0] - pos_counts
    pos_weight = neg_counts / np.maximum(pos_counts, 1.0)
    pos_weight = np.clip(pos_weight, 1.0, 50.0).astype(np.float32)
    return pos_weight


def train():
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"[Train] Device: {device}")
    print("[Train] Loading PKU-SafeRLHF dataset...")
    wandb_logger = WandbLogger()
    wandb_logger.init_run(
        job_type="discriminator_train",
        name="discriminator-train",
        config_dict={
            "model_checkpoint": MODEL_CHECKPOINT,
            "max_len": MAX_LEN,
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "learning_rate": LR,
            "warmup_ratio": WARMUP_RATIO,
            "num_risk_labels": NUM_RISK_LABELS,
            "max_train_samples": MAX_TRAIN_SAMPLES,
            "max_eval_samples": MAX_EVAL_SAMPLES,
        },
    )

    raw = load_local_pku_dataset()
    train_split = raw["train"]
    eval_split = raw.get("test", raw.get("validation", raw["train"].select(range(min(5000, len(raw["train"]))))))  # noqa: E501

    print(f"[Train] Raw train size: {len(train_split)}, eval size: {len(eval_split)}")

    train_samples = extract_samples_from_dataset(train_split, MAX_TRAIN_SAMPLES)
    eval_samples = extract_samples_from_dataset(eval_split, MAX_EVAL_SAMPLES)

    train_dist = Counter()
    for _, labels in train_samples:
        for idx, value in enumerate(labels):
            if value > 0:
                train_dist[idx] += 1
    print(f"[Train] Positive label counts: {{ {', '.join(f'{RISK_LABEL_NAMES[k]}: {train_dist[k]}' for k in sorted(train_dist))} }}")
    pos_weight = compute_pos_weight(train_samples)
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32, device=device)
    print(
        "[Train] Pos weights: { "
        + ", ".join(f"{RISK_LABEL_NAMES[i]}: {pos_weight[i]:.2f}" for i in range(NUM_RISK_LABELS))
        + " }"
    )

    print(f"[Train] Loading tokenizer & model: {MODEL_CHECKPOINT}")
    model_path = resolve_model_path(MODEL_CHECKPOINT)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=NUM_RISK_LABELS,
        problem_type="multi_label_classification",
    ).to(device)

    train_ds = SafetyDataset(train_samples, tokenizer, MAX_LEN)
    eval_ds = SafetyDataset(eval_samples, tokenizer, MAX_LEN)

    train_generator = torch.Generator().manual_seed(Config.SEED)
    eval_generator = torch.Generator().manual_seed(Config.SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, generator=train_generator)
    eval_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, generator=eval_generator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_eval_micro_acc = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_count = 0

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                labels,
                pos_weight=pos_weight_tensor,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * labels.size(0)
            total_count += labels.size(0)

            if (step + 1) % 100 == 0:
                print(f"  Epoch {epoch} | Step {step+1}/{len(train_loader)} | Loss={total_loss / total_count:.4f}")

        train_loss = total_loss / max(total_count, 1)
        print(f"[Epoch {epoch}] Train Loss={train_loss:.4f}")

        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.sigmoid(outputs.logits)

                all_probs.extend(probs.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        exact_match, micro_acc, preds = multilabel_metrics(all_labels, all_probs)
        print(f"[Epoch {epoch}] Eval ExactMatch={exact_match:.4f} | MicroAcc={micro_acc:.4f}")
        wandb_logger.log(
            {
                "discriminator/train_loss": train_loss,
                "discriminator/eval_exact_match": exact_match,
                "discriminator/eval_micro_acc": micro_acc,
                "discriminator/epoch": epoch,
            },
            step=epoch,
        )
        print(
            classification_report(
                np.asarray(all_labels, dtype=int),
                preds,
                target_names=RISK_LABEL_NAMES,
                zero_division=0,
            )
        )

        if micro_acc > best_eval_micro_acc:
            best_eval_micro_acc = micro_acc
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            np.save(os.path.join(OUTPUT_DIR, POS_WEIGHT_FILE), pos_weight)
            print(f"  Saved best model (micro_acc={best_eval_micro_acc:.4f}) -> {OUTPUT_DIR}")

    wandb_logger.summary_update(
        {
            "summary/best_eval_micro_acc": best_eval_micro_acc,
        }
    )
    wandb_logger.finish()

    print(f"\n[Train] Done. Best eval micro_acc = {best_eval_micro_acc:.4f}")
    print(f"[Train] Model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    train()
