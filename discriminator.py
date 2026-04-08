import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import Config
from hf_utils import resolve_model_path
from risk_taxonomy import (
    NUM_RISK_LABELS,
    RISK_LABEL_NAMES,
    aggregate_risk_score,
    empty_multihot,
)


FINETUNED_DIR = "./safety_discriminator_multilabel"
FALLBACK_MODEL = "distilbert-base-uncased"
POS_WEIGHT_FILE = "pos_weight.npy"


class SafetyDiscriminator:
    """
    Multi-label risk discriminator trained on PKU-SafeRLHF harm categories.

    The model outputs one probability per risk type, then aggregates them into
    a normalized scalar risk score for the scheduler.
    """

    def __init__(self, use_mc_dropout: bool = True, mc_samples: int = 10):
        self.use_mc_dropout = use_mc_dropout
        self.mc_samples = mc_samples

        if os.path.isdir(FINETUNED_DIR):
            model_path = FINETUNED_DIR
            print(f"[Discriminator] Loading fine-tuned model from: {model_path}")
        else:
            model_path = FALLBACK_MODEL
            print(f"[Discriminator] Fine-tuned model not found; loading base model: {model_path}")
            print("  Run train_safety_discriminator.py first for best results.")

        resolved_model_path = resolve_model_path(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            resolved_model_path,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            resolved_model_path,
            num_labels=NUM_RISK_LABELS,
            problem_type="multi_label_classification",
        ).to(Config.DEVICE)
        self.model.eval()

        self.pos_weight = self._load_pos_weight(resolved_model_path)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-6)
        self.replay_buffer = deque(maxlen=1000)
        self._seed_buffer()

    def _seed_buffer(self):
        seed_data = [
            ("The weather is nice today", empty_multihot()),
            ("Hello, how are you?", empty_multihot()),
            ("Have a great day!", empty_multihot()),
            ("I hate all of you, go die", self._single_hot(1)),
            ("Women are inferior to men", self._single_hot(2)),
            ("How to make a bomb at home", self._single_hot(11)),
            ("I want to hurt myself", self._single_hot(16)),
            ("Send me your bank details now", self._single_hot(17)),
            ("How to do a phishing scam", self._single_hot(13)),
            ("How to hack into someone's email", self._single_hot(7)),
            ("Steps to pirate software illegally", self._single_hot(4)),
        ]
        for text, labels in seed_data:
            self.replay_buffer.append((text, labels.astype(np.float32)))

    def _single_hot(self, idx: int) -> np.ndarray:
        labels = empty_multihot()
        labels[idx] = 1.0
        return labels

    def predict(self, text: str):
        """
        Returns
        -------
        class_probs   : np.ndarray, shape (NUM_RISK_LABELS,)
        predicted_cls : int
        uncertainty   : float in [0, 1]
        """
        if self.use_mc_dropout:
            return self._predict_mc_dropout(text)
        return self._predict_deterministic(text)

    def risk_score(self, text: str) -> float:
        class_probs, _, _ = self.predict(text)
        return aggregate_risk_score(class_probs)

    def _predict_deterministic(self, text: str):
        inputs = self._tokenize([text])
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.sigmoid(logits)[0]

        class_probs = probs.cpu().numpy()
        predicted_cls = int(class_probs.argmax())
        uncertainty = self._binary_entropy_uncertainty(class_probs)
        return class_probs, predicted_cls, uncertainty

    def _predict_mc_dropout(self, text: str):
        inputs = self._tokenize([text])
        self._enable_dropout()

        all_probs = []
        with torch.no_grad():
            for _ in range(self.mc_samples):
                logits = self.model(**inputs).logits
                probs = torch.sigmoid(logits)[0]
                all_probs.append(probs.cpu().numpy())

        self.model.eval()

        all_probs_arr = np.stack(all_probs, axis=0)
        mean_probs = all_probs_arr.mean(axis=0)
        predicted_cls = int(mean_probs.argmax())

        entropy_term = self._binary_entropy(mean_probs)
        variance_term = float(all_probs_arr.std(axis=0).mean())
        uncertainty = float(0.5 * entropy_term + 0.5 * variance_term)
        uncertainty = min(1.0, max(0.0, uncertainty))

        return mean_probs, predicted_cls, uncertainty

    def update_with_replay(self, new_text: str, human_label, batch_size: int = 8):
        if human_label == 1:
            labels = empty_multihot()
        elif human_label == -1:
            raise ValueError(
                "Unsafe replay updates require a 19-dim multi-hot label vector. "
                "Pass the human reviewer risk_multihot instead of bare -1."
            )
        else:
            label_arr = np.asarray(human_label, dtype=np.float32)
            if label_arr.shape != (NUM_RISK_LABELS,):
                raise ValueError("Expected human_label to be 1/-1 or a 19-dim multi-hot vector.")
            labels = label_arr

        self.replay_buffer.append((new_text, labels.astype(np.float32)))

        buf_list = list(self.replay_buffer)
        if len(buf_list) < batch_size:
            batch_data = buf_list
        else:
            batch_data = random.sample(buf_list, batch_size)

        if not any(text == new_text for text, _ in batch_data):
            batch_data[0] = (new_text, labels.astype(np.float32))

        texts = [item[0] for item in batch_data]
        label_tensor = torch.tensor(
            np.stack([item[1] for item in batch_data], axis=0),
            dtype=torch.float32,
            device=Config.DEVICE,
        )

        self.model.train()
        inputs = self._tokenize(texts)
        logits = self.model(**inputs).logits
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            label_tensor,
            pos_weight=self.pos_weight,
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        self.model.eval()
        return loss.item()

    def _tokenize(self, texts: list):
        return self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256,
        ).to(Config.DEVICE)

    def _enable_dropout(self):
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.train()

    def _load_pos_weight(self, resolved_model_path: str):
        pos_weight_path = os.path.join(resolved_model_path, POS_WEIGHT_FILE)
        if os.path.isfile(pos_weight_path):
            values = np.load(pos_weight_path).astype(np.float32)
        else:
            values = np.ones(NUM_RISK_LABELS, dtype=np.float32)
        return torch.tensor(values, dtype=torch.float32, device=Config.DEVICE)

    def label_name(self, class_idx: int) -> str:
        if 0 <= class_idx < len(RISK_LABEL_NAMES):
            return RISK_LABEL_NAMES[class_idx]
        return f"Unknown({class_idx})"

    def top_risks(self, class_probs, k: int = 3):
        probs = np.asarray(class_probs)
        top_indices = probs.argsort()[::-1][:k]
        return [(self.label_name(int(idx)), float(probs[idx])) for idx in top_indices]

    def _binary_entropy(self, probs: np.ndarray) -> float:
        eps = 1e-8
        p = np.clip(probs, eps, 1.0 - eps)
        entropy = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
        max_entropy = np.log(2.0)
        return float(entropy.mean() / max_entropy)

    def _binary_entropy_uncertainty(self, probs: np.ndarray) -> float:
        return self._binary_entropy(np.asarray(probs, dtype=np.float32))
