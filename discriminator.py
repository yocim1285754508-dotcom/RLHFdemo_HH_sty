# discriminator.py
import torch
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import Config

class SafetyDiscriminator:
    def __init__(self, use_mc_dropout=True):
        self.use_mc_dropout = use_mc_dropout
        
        # === 模型加载 ===
        self.model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        print(f"[Discriminator] Loading semantic model: {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(Config.DEVICE)
        
        # === 优化器 ===
        # 学习率要非常低，因为我们是在微调一个已经很成熟的模型
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-6)
        
        # === 核心升级: Experience Replay Buffer ===
        # maxlen=1000: 最多记住最近 1000 条，旧的会自动挤出去
        self.replay_buffer = deque(maxlen=1000)
        
        # 预先填入一些基础样本，防止一开始 Buffer 为空导致报错
        # (在实际工程中，这里应该加载一个小的验证集)
        self._seed_buffer()

    def _seed_buffer(self):
        """给记忆池一点初始数据，防止冷启动问题"""
        initial_data = [
            ("I want to kill everyone", 0), # 0 = Unsafe
            ("How to make a bomb", 0),
            ("The weather is nice", 1),     # 1 = Safe
            ("Hello world", 1),
            ("I hate you", 0),
            ("Have a nice day", 1)
        ]
        for text, label in initial_data:
            self.replay_buffer.append((text, label))

    def predict(self, text):
        """保持原有的预测逻辑不变"""
        if self.use_mc_dropout:
            return self._predict_with_mc_dropout(text)
        else:
            return self._predict_deterministic(text)

    # ... (这里保留你之前的 _predict_deterministic 和 _predict_with_mc_dropout 代码) ...
    # 为了节省篇幅，假设你已经保留了这两个函数
    # ... 
    
    def _predict_deterministic(self, text):
        inputs = self._tokenize([text]) # 注意：tokenize 接受列表
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
        risk_score = probs[0][0].item()
        uncertainty = 1.0 - abs(risk_score - 0.5) * 2
        return risk_score, uncertainty

    def _predict_with_mc_dropout(self, text, num_samples=5):
        inputs = self._tokenize([text])
        self.model.train() # 开启 Dropout
        batch_risks = []
        with torch.no_grad():
            for _ in range(num_samples):
                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)
                batch_risks.append(probs[0][0].item())
        self.model.eval()
        avg_risk = np.mean(batch_risks)
        uncertainty = np.std(batch_risks)
        uncertainty_score = min(1.0, uncertainty * 5)
        return avg_risk, uncertainty_score

    def update_with_replay(self, new_text, human_label, batch_size=4):
        """
        === 核心升级: 带回放的更新 ===
        不仅仅学这一条，而是混合记忆一起学
        """
        # 1. 把新经验存入 Buffer
        # human_label: 1 (Safe) -> target 1; -1 (Unsafe) -> target 0
        target_label = 1 if human_label == 1 else 0
        self.replay_buffer.append((new_text, target_label))
        
        # 2. 采样 (Sampling)
        # 如果数据不够一个 batch，就全拿出来；否则随机抽 batch_size 个
        if len(self.replay_buffer) < batch_size:
            batch_data = list(self.replay_buffer)
        else:
            batch_data = random.sample(self.replay_buffer, batch_size)
            
        # 确保当前这条新数据一定在 batch 里（以此保证对新错误的修正能力）
        # 这是一个 Trick：Replay 负责稳，强制加入新数据负责准
        if (new_text, target_label) not in batch_data:
            batch_data[0] = (new_text, target_label)
            
        # 3. 解包数据
        texts = [item[0] for item in batch_data]
        labels = torch.tensor([item[1] for item in batch_data]).to(Config.DEVICE)
        
        # 4. 训练步 (Training Step)
        self.model.train()
        
        # Tokenize 一个 Batch
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(Config.DEVICE)
        
        outputs = self.model(**inputs, labels=labels)
        loss = outputs.loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.model.eval() # 恢复评估模式
        
        return loss.item()

    def _tokenize(self, texts):
        # 辅助函数：处理列表输入
        return self.tokenizer(
            texts, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=128
        ).to(Config.DEVICE)