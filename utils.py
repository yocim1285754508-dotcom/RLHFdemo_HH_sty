# utils.py
import json
from pathlib import Path
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

def save_metrics(metrics, output_path):
    """Dump metrics or evaluation artifacts to disk as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Metrics] Saved {output_path}")

def plot_metric(metric_points, metric_key, output_path, ylabel=None, title=None):
    """Render a simple line plot for a metric across training steps."""
    if not metric_points:
        print(f"[Plot] Skip plotting {metric_key}: no data")
        return
    sorted_points = sorted(
        metric_points,
        key=lambda item: (item.get("epoch", 0), item.get("step", 0))
    )
    y_values = [item.get(metric_key) for item in sorted_points if metric_key in item]
    if not y_values:
        print(f"[Plot] Skip plotting {metric_key}: not found in metrics")
        return
    x_values = list(range(len(y_values)))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    plt.plot(x_values, y_values, label=metric_key)
    plt.xlabel("Step")
    plt.ylabel(ylabel or metric_key)
    plt.title(title or f"{metric_key} over time")
    plt.legend()
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"[Plot] Wrote {output_path}")
