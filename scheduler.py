from config import Config


class FeedbackScheduler:
    def __init__(self):
        self.human_budget = Config.HUMAN_BUGET

    def decide(self, risk_score, uncertainty):
        """
        Route to human review when the sample is either risk-ambiguous or too
        uncertain, as long as human review budget remains.
        """
        is_ambiguous = (
            Config.RISK_THRESHOLD_LOW <= risk_score <= Config.RISK_THRESHOLD_HIGH
        )
        is_uncertain = uncertainty > Config.UNCERTAINTY_THRESHOLD

        if (is_ambiguous or is_uncertain) and self.human_budget > 0:
            self.human_budget -= 1
            return "HUMAN", True
        return "AI_AUTO", False

    def get_reward_source(self, source_type, ai_score, human_label=None):
        if source_type == "HUMAN" and human_label is not None:
            return human_label * Config.HUMAN_WEIGHT

        return (1.0 - 2 * ai_score) * Config.SAFETY_WEIGHT
