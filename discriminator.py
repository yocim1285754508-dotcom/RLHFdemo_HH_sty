# discriminator.py
"""
多分类安全鉴别器（MC Dropout 不确定性估计）

标签体系（5 类，与 train_safety_discriminator.py 保持一致）:
  0 - Safe              安全内容
  1 - Offensive         攻击性/仇恨言论
  2 - Physical Harm     身体/心理伤害
  3 - Fraud/Financial   欺诈/经济伤害
  4 - Privacy/Legal     隐私/违法

predict() 返回:
  class_probs  (np.ndarray, shape=[NUM_LABELS])  各类别概率均值
  predicted_cls (int)                            预测类别
  uncertainty   (float, 0~1)                     预测不确定度（来自 MC Dropout）
"""

import torch
import torch.nn.functional as F
import numpy as np
import random
import os
from collections import deque
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import Config

# ─────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────
NUM_LABELS   = 5
LABEL_NAMES  = ["Safe", "Offensive", "Physical Harm", "Fraud/Financial", "Privacy/Legal"]
SAFE_LABEL   = 0   # 安全类别的整数索引

# 训练脚本保存的模型路径；若不存在则回退到 HuggingFace 基础模型
FINETUNED_DIR   = "./safety_discriminator_multiclass"
FALLBACK_MODEL  = "distilbert-base-uncased"


class SafetyDiscriminator:
    """多分类安全鉴别器，支持 MC Dropout 不确定性估计。"""

    def __init__(self, use_mc_dropout: bool = True, mc_samples: int = 10):
        self.use_mc_dropout = use_mc_dropout
        self.mc_samples     = mc_samples

        # ── 加载模型 ──────────────────────────────
        if os.path.isdir(FINETUNED_DIR):
            model_path = FINETUNED_DIR
            print(f"[Discriminator] Loading fine-tuned model from: {model_path}")
        else:
            model_path = FALLBACK_MODEL
            print(f"[Discriminator] Fine-tuned model not found; loading base model: {model_path}")
            print(f"  → Run train_safety_discriminator.py first for best results.")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=NUM_LABELS,
        ).to(Config.DEVICE)
        self.model.eval()

        # ── 优化器（在线微调用）───────────────────
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-6)

        # ── Experience Replay Buffer ──────────────
        # 每条记录: (text: str, label: int)，label 为 0~4 的整数
        self.replay_buffer = deque(maxlen=1000)
        self._seed_buffer()

    # ─────────────────────────────────────────────
    # 初始化数据
    # ─────────────────────────────────────────────
    def _seed_buffer(self):
        """给 Replay Buffer 注入初始种子数据，防止冷启动时 buffer 为空。"""
        seed_data = [
            ("The weather is nice today",      0),  # Safe
            ("Hello, how are you?",             0),  # Safe
            ("Have a great day!",               0),  # Safe
            ("I hate all of you, go die",       1),  # Offensive
            ("Women are inferior to men",       1),  # Offensive
            ("How to make a bomb at home",      2),  # Physical Harm
            ("I want to hurt myself",           2),  # Physical Harm
            ("Send me your bank details now",   3),  # Fraud/Financial
            ("How to do a phishing scam",       3),  # Fraud/Financial
            ("How to hack into someone's email",4),  # Privacy/Legal
            ("Steps to pirate software illegally",4),# Privacy/Legal
        ]
        for text, label in seed_data:
            self.replay_buffer.append((text, label))

    # ─────────────────────────────────────────────
    # 公共预测接口
    # ─────────────────────────────────────────────
    def predict(self, text: str):
        """
        对输入文本进行安全评估。

        Returns
        -------
        class_probs   : np.ndarray, shape (NUM_LABELS,)  各类别概率（均值）
        predicted_cls : int                              概率最大的类别索引
        uncertainty   : float  ∈ [0, 1]                  预测不确定度
        """
        if self.use_mc_dropout:
            return self._predict_mc_dropout(text)
        else:
            return self._predict_deterministic(text)

    def is_safe(self, text: str) -> bool:
        """快捷接口：判断文本是否安全（predicted_cls == SAFE_LABEL）。"""
        _, predicted_cls, _ = self.predict(text)
        return predicted_cls == SAFE_LABEL

    def risk_score(self, text: str) -> float:
        """
        快捷接口：返回 [0,1] 风险分（与旧代码兼容）。
        risk_score = 1 - P(Safe)
        """
        class_probs, _, _ = self.predict(text)
        return float(1.0 - class_probs[SAFE_LABEL])

    # ─────────────────────────────────────────────
    # 确定性预测
    # ─────────────────────────────────────────────
    def _predict_deterministic(self, text: str):
        inputs = self._tokenize([text])
        with torch.no_grad():
            logits = self.model(**inputs).logits          # (1, NUM_LABELS)
            probs  = F.softmax(logits, dim=-1)[0]         # (NUM_LABELS,)

        class_probs   = probs.cpu().numpy()
        predicted_cls = int(class_probs.argmax())

        # 不确定度：用最大概率的反向度量
        max_prob    = class_probs.max()
        uncertainty = float(1.0 - max_prob)               # 0 = 完全确定，1 = 完全不确定

        return class_probs, predicted_cls, uncertainty

    # ─────────────────────────────────────────────
    # MC Dropout 预测（多分类版）
    # ─────────────────────────────────────────────
    def _predict_mc_dropout(self, text: str):
        """
        MC Dropout 多分类不确定性估计。

        在 train() 模式下进行 mc_samples 次前向传播，
        利用 Dropout 的随机性采样后验概率分布。

        不确定度指标：
          - mean_entropy  : 平均预测熵（主要指标）
          - prob_std_mean : 各类别概率标准差的均值（辅助）
        最终 uncertainty = mean_entropy / log(NUM_LABELS)  ∈ [0,1]
        """
        inputs = self._tokenize([text])

        # 开启 Dropout（保持 BatchNorm 在 eval 状态）
        self._enable_dropout()

        all_probs = []
        with torch.no_grad():
            for _ in range(self.mc_samples):
                logits = self.model(**inputs).logits          # (1, NUM_LABELS)
                probs  = F.softmax(logits, dim=-1)[0]         # (NUM_LABELS,)
                all_probs.append(probs.cpu().numpy())

        self.model.eval()

        all_probs_arr = np.stack(all_probs, axis=0)           # (mc_samples, NUM_LABELS)

        # 均值概率
        mean_probs    = all_probs_arr.mean(axis=0)            # (NUM_LABELS,)
        predicted_cls = int(mean_probs.argmax())

        # 预测熵作为不确定度（归一化到 [0,1]）
        eps          = 1e-8
        entropy      = -np.sum(mean_probs * np.log(mean_probs + eps))
        max_entropy  = np.log(NUM_LABELS)                     # 均匀分布时的最大熵
        uncertainty  = float(entropy / max_entropy)

        return mean_probs, predicted_cls, uncertainty

    # ─────────────────────────────────────────────
    # 在线微调（带 Replay Buffer）
    # ─────────────────────────────────────────────
    def update_with_replay(self, new_text: str, human_label: int, batch_size: int = 8):
        """
        接收一条人工标注数据并触发一次在线微调。

        Parameters
        ----------
        new_text    : 需要学习的文本
        human_label : 正确标签，整数 0~4 或旧兼容格式:
                      -1 → 1 (Offensive), 1 → 0 (Safe)
        batch_size  : Replay 采样大小

        Returns
        -------
        loss : float  本次训练的交叉熵 loss
        """
        # 兼容旧版二分类标签（1=Safe, -1=Unsafe）
        if human_label == 1:
            label = 0   # Safe
        elif human_label == -1:
            label = 1   # Offensive（默认归为 offensive）
        else:
            label = int(human_label)  # 已经是多分类标签

        # 存入 Buffer
        self.replay_buffer.append((new_text, label))

        # 采样
        buf_list = list(self.replay_buffer)
        if len(buf_list) < batch_size:
            batch_data = buf_list
        else:
            batch_data = random.sample(buf_list, batch_size)

        # 确保新样本在 batch 中
        if (new_text, label) not in batch_data:
            batch_data[0] = (new_text, label)

        texts  = [item[0] for item in batch_data]
        labels = torch.tensor([item[1] for item in batch_data], dtype=torch.long).to(Config.DEVICE)

        # 训练步
        self.model.train()
        inputs  = self._tokenize(texts)
        outputs = self.model(**inputs, labels=labels)
        loss    = outputs.loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        self.model.eval()
        return loss.item()

    # ─────────────────────────────────────────────
    # 工具函数
    # ─────────────────────────────────────────────
    def _tokenize(self, texts: list):
        return self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256,
        ).to(Config.DEVICE)

    def _enable_dropout(self):
        """只开启 Dropout 层，保持 LayerNorm / BatchNorm 在 eval 模式。"""
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.train()

    def label_name(self, class_idx: int) -> str:
        """返回类别名称字符串。"""
        if 0 <= class_idx < len(LABEL_NAMES):
            return LABEL_NAMES[class_idx]
        return f"Unknown({class_idx})"
