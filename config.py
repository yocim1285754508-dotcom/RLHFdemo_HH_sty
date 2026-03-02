# config.py
import torch

class Config:
    # 模型配置
    MODEL_NAME = "sshleifer/tiny-gpt2"  # 极小模型，适合无 GPU 的 Demo
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42

    # 训练超参数（为 8GB Mac 缩减）
    LEARNING_RATE = 5e-6
    BATCH_SIZE = 1   # 维持单条样本，避免显存峰值
    EPOCHS = 2       # Demo 仅跑 2 个 epoch
    MAX_GEN_LEN = 24 # 限制生成长度以节省显存/内存
    HH_NUM_SAMPLES = 5 # 只取少量 prompt 作为演示

    # === Method 核心设计：调度器阈值 ===
    # 风险分数范围 [0, 1]
    # 如果分数在 [LOW, HIGH] 之间，视为“模糊”，触发人工
    RISK_THRESHOLD_LOW = 0.3
    RISK_THRESHOLD_HIGH = 0.8
    
    # === 奖励聚合参数 ===
    KL_COEF = 0.1       # KL 散度惩罚系数 (Beta)
    SAFETY_WEIGHT = 2.0 # 安全奖励的权重
    HUMAN_WEIGHT = 5.0  # 人类反馈的权重 (Ground Truth 拥有最高权重)

    # 人类预算
    HUMAN_BUGET = 100 # 人类预算的次数以约束计算的代价
