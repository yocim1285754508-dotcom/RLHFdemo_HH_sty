# utils.py
import random
import numpy as np
import torch
from config import Config

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_human_input(prompt, response):
    """模拟人类在环的输入接口"""
    print(f"\n[!!! HUMAN INTERVENTION REQUIRED !!!]")
    print(f"Prompt: {prompt}")
    print(f"Model Generated: {response}")
    print("Is this safe? (Input 1 for Safe, -1 for Unsafe): ", end="")
    try:
        # 在实际 Demo 运行时，这里可以取消注释接收键盘输入
        # feedback = int(input())
        feedback = -1 # 演示默认认为不安全
        print(f"Human labeled as: {feedback}")
        return feedback
    except:
        return -1