import random

import numpy as np
import torch

from config import Config
from openai_evaluator import OpenAIHumanReviewer


def set_seed(seed=Config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def get_human_input(prompt, response):
    reviewer = OpenAIHumanReviewer()
    print("\n[!!! HUMAN INTERVENTION REQUIRED !!!]")
    print(f"Prompt: {prompt}")
    print(f"Model Generated: {response}")

    try:
        result = reviewer.evaluate(prompt, response)
        feedback = {
            "label": int(result["label"]),
            "risk_score": float(result["risk_score"]),
            "reasoning": result["reasoning"],
            "risk_multihot": result["risk_multihot"],
        }
        print(f"Simulated human label: {feedback['label']}")
        print(f"Simulated human risk: {float(result['risk_score']):.2f}")
        print(f"Simulated human reasoning: {result['reasoning']}")
        return feedback
    except Exception as exc:
        print(f"Simulated human review failed, fallback safe-zero label. Error: {exc}")
        return {
            "label": 1,
            "risk_score": 0.0,
            "reasoning": f"Fallback because simulated human review failed: {exc}",
            "risk_multihot": [0] * 19,
        }
