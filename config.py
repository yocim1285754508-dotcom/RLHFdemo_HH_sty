import torch


class Config:
    MODEL_NAME = "gpt2"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42

    LEARNING_RATE = 1e-5
    BATCH_SIZE = 1
    EPOCHS = 10
    MAX_GEN_LEN = 50

    RISK_THRESHOLD_LOW = 0.3
    RISK_THRESHOLD_HIGH = 0.8
    UNCERTAINTY_THRESHOLD = 0.15

    KL_COEF = 0.1
    RETURN_GAMMA = 1.0
    SAFETY_WEIGHT = 2.0
    HUMAN_WEIGHT = 5.0

    HUMAN_BUGET = 100
