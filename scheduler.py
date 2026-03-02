# scheduler.py
from config import Config

class FeedbackScheduler:
    def __init__(self):
        self.human_budget = Config.HUMAN_BUGET  # 模拟：我们只有 5 次请专家的机会
    
    def decide(self, risk_score, uncertainty):
        """
        Method 核心逻辑：基于不确定性的路由
        """
        # 硬规则：判断是否在模糊区间
        is_ambiguous = (Config.RISK_THRESHOLD_LOW <= risk_score <= Config.RISK_THRESHOLD_HIGH)
        
        # 只有在模糊且还有预算时，才请求人类
        if is_ambiguous and self.human_budget > 0:
            self.human_budget -= 1
            return "HUMAN", True
        else:
            return "AI_AUTO", False

    def get_reward_source(self, source_type, ai_score, human_label=None):
        """
        Reward Aggregator 的一部分逻辑：决定信谁
        """
        if source_type == "HUMAN" and human_label is not None:
            # 人类拥有最高权限 (Ground Truth)
            # human_label: +1 (Safe), -1 (Unsafe)
            return human_label * Config.HUMAN_WEIGHT
        else:
            # AI 自动打分
            # 将 [0,1] 的分数映射到 [-1, 1] 的奖励
            # risk 0.9 -> reward -0.9
            # risk 0.1 -> reward +0.9
            return (1.0 - 2 * ai_score) * Config.SAFETY_WEIGHT