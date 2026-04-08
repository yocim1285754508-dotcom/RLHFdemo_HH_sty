import numpy as np


RISK_LABEL_NAMES = [
    "Endangering National Security",
    "Insulting Behavior",
    "Discriminatory Behavior",
    "Endangering Public Health",
    "Copyright Issues",
    "Violence",
    "Drugs",
    "Privacy Violation",
    "Economic Crime",
    "Mental Manipulation",
    "Human Trafficking",
    "Physical Harm",
    "Sexual Content",
    "Cybercrime",
    "Disrupting Public Order",
    "Environmental Damage",
    "Psychological Harm",
    "White-Collar Crime",
    "Animal Abuse",
]

NUM_RISK_LABELS = len(RISK_LABEL_NAMES)
DEFAULT_RISK_WEIGHTS = np.ones(NUM_RISK_LABELS, dtype=np.float32)
TOXIC_RISK_INDICES = [1, 2, 5, 12, 16]
RISK_AGGREGATION_TOP_K = 3
RISK_SCORE_MARGIN = 0.35
RISK_SCORE_POWER = 0.5

PKU_HARM_ALIASES = {
    "s1": 0,
    "endangering_national_security": 0,
    "endangering national security": 0,
    "s2": 1,
    "insulting_behavior": 1,
    "insulting behavior": 1,
    "offensiveness": 1,
    "s3": 2,
    "discriminatory_behavior": 2,
    "discriminatory behavior": 2,
    "discrimination": 2,
    "hate_speech": 2,
    "s4": 3,
    "endangering_public_health": 3,
    "endangering public health": 3,
    "public_health": 3,
    "s5": 4,
    "copyright_issues": 4,
    "copyright issues": 4,
    "copyright": 4,
    "s6": 5,
    "violence": 5,
    "s7": 6,
    "drugs": 6,
    "s8": 7,
    "privacy_violation": 7,
    "privacy violation": 7,
    "privacy": 7,
    "s9": 8,
    "economic_crime": 8,
    "economic crime": 8,
    "financial_harm": 8,
    "s10": 9,
    "mental_manipulation": 9,
    "mental manipulation": 9,
    "s11": 10,
    "human_trafficking": 10,
    "human trafficking": 10,
    "s12": 11,
    "physical_harm": 11,
    "physical harm": 11,
    "physical_safety": 11,
    "s13": 12,
    "sexual_content": 12,
    "sexual content": 12,
    "s14": 13,
    "cybercrime": 13,
    "s15": 14,
    "disrupting_public_order": 14,
    "disrupting public order": 14,
    "s16": 15,
    "environmental_damage": 15,
    "environmental damage": 15,
    "s17": 16,
    "psychological_harm": 16,
    "psychological harm": 16,
    "self_harm": 16,
    "s18": 17,
    "white-collar_crime": 17,
    "white-collar crime": 17,
    "fraud": 17,
    "deception": 17,
    "s19": 18,
    "animal_abuse": 18,
    "animal abuse": 18,
}


def normalize_harm_key(key: str) -> str:
    return key.strip().lower().replace("-", "_")


def empty_multihot() -> np.ndarray:
    return np.zeros(NUM_RISK_LABELS, dtype=np.float32)


def harm_types_to_multihot(harm_types) -> np.ndarray:
    labels = empty_multihot()
    if not harm_types:
        return labels

    for raw_key, value in harm_types.items():
        if not value:
            continue
        key = normalize_harm_key(str(raw_key))
        idx = PKU_HARM_ALIASES.get(key)
        if idx is not None:
            labels[idx] = 1.0
    return labels


def aggregate_risk_score(class_probs, risk_weights=None) -> float:
    probs = np.asarray(class_probs, dtype=np.float32)
    weights = DEFAULT_RISK_WEIGHTS if risk_weights is None else np.asarray(risk_weights, dtype=np.float32)
    if probs.size == 0:
        return 0.0

    k = min(RISK_AGGREGATION_TOP_K, probs.size)
    top_indices = np.argsort(probs)[-k:]
    top_probs = probs[top_indices]
    top_weights = weights[top_indices]

    # Ignore weak activations and amplify confident risky categories so
    # dangerous samples are not washed out by many low-probability classes.
    margin_adjusted = np.clip((top_probs - RISK_SCORE_MARGIN) / (1.0 - RISK_SCORE_MARGIN), 0.0, 1.0)
    amplified = np.power(margin_adjusted, RISK_SCORE_POWER)

    denom = float(np.sum(top_weights))
    if denom <= 0:
        return float(np.mean(amplified))
    return float(np.sum(amplified * top_weights) / denom)


def aggregate_toxicity_score(class_probs) -> float:
    probs = np.asarray(class_probs, dtype=np.float32)
    if len(TOXIC_RISK_INDICES) == 0:
        return float(probs.mean())
    return float(probs[TOXIC_RISK_INDICES].mean())
