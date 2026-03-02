# config.py
import torch

class Config:
    # 模型配置
    MODEL_NAME = "gpt2"  # 使用轻量级模型演示，可换成 distilgpt2 或其它
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42

    # 训练超参数
    LEARNING_RATE = 1e-5
    BATCH_SIZE = 1  # 演示用，工业级可增大
    EPOCHS = 10     # 演示轮数
    MAX_GEN_LEN = 50 # 生成的最大长度

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