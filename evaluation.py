from collections import Counter

import numpy as np

from config import Config
from discriminator import SafetyDiscriminator
from openai_evaluator import OpenAIQualityEvaluator
from risk_taxonomy import aggregate_toxicity_score
from wandb_logger import WandbLogger
from weave_support import weave_op


def simple_tokenize(text: str):
    return text.lower().split()


def ngram_counts(tokens, n):
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def sentence_bleu4(reference: str, candidate: str) -> float:
    ref_tokens = simple_tokenize(reference)
    cand_tokens = simple_tokenize(candidate)
    if not cand_tokens:
        return 0.0

    precisions = []
    for n in range(1, 5):
        cand_counts = ngram_counts(cand_tokens, n)
        ref_counts = ngram_counts(ref_tokens, n)
        if not cand_counts:
            precisions.append(0.0)
            continue
        overlap = sum(min(count, ref_counts[gram]) for gram, count in cand_counts.items())
        precisions.append((overlap + 1.0) / (sum(cand_counts.values()) + 1.0))

    ref_len = len(ref_tokens)
    cand_len = len(cand_tokens)
    if cand_len == 0:
        return 0.0
    if cand_len > ref_len:
        brevity_penalty = 1.0
    else:
        brevity_penalty = np.exp(1.0 - (ref_len / max(cand_len, 1)))

    return float(brevity_penalty * np.exp(np.mean(np.log(np.maximum(precisions, 1e-12)))))


def lcs_length(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def rouge_l(reference: str, candidate: str) -> float:
    ref_tokens = simple_tokenize(reference)
    cand_tokens = simple_tokenize(candidate)
    if not ref_tokens or not cand_tokens:
        return 0.0

    lcs = lcs_length(ref_tokens, cand_tokens)
    precision = lcs / len(cand_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    beta = 1.2
    return float(((1 + beta**2) * precision * recall) / (recall + beta**2 * precision))


def evaluate_safety_metrics(prompts, responses, discriminator=None, violation_threshold=None):
    if discriminator is None:
        discriminator = SafetyDiscriminator()

    threshold = Config.EVAL_VIOLATION_THRESHOLD if violation_threshold is None else violation_threshold
    risk_scores = []
    toxicity_scores = []
    uncertainties = []
    violation_flags = []

    for response in responses:
        class_probs, _, uncertainty = discriminator.predict(response)
        risk_score = discriminator.risk_score(response)
        toxicity_score = aggregate_toxicity_score(class_probs)

        risk_scores.append(risk_score)
        toxicity_scores.append(toxicity_score)
        uncertainties.append(float(uncertainty))
        violation_flags.append(float(risk_score >= threshold))

    return {
        "violation_rate": float(np.mean(violation_flags)) if violation_flags else 0.0,
        "avg_risk_score": float(np.mean(risk_scores)) if risk_scores else 0.0,
        "avg_toxicity_score": float(np.mean(toxicity_scores)) if toxicity_scores else 0.0,
        "avg_uncertainty": float(np.mean(uncertainties)) if uncertainties else 0.0,
    }


def evaluate_utility_metrics(responses, references):
    if not references or len(responses) != len(references):
        return {
            "bleu4": None,
            "rouge_l": None,
        }

    bleu_scores = [sentence_bleu4(ref, resp) for resp, ref in zip(responses, references)]
    rouge_scores = [rouge_l(ref, resp) for resp, ref in zip(responses, references)]

    return {
        "bleu4": float(np.mean(bleu_scores)) if bleu_scores else 0.0,
        "rouge_l": float(np.mean(rouge_scores)) if rouge_scores else 0.0,
    }


def evaluate_quality_metrics(prompts, responses, references=None, quality_evaluator=None):
    if quality_evaluator is None:
        quality_evaluator = OpenAIQualityEvaluator()

    usefulness_scores = []
    naturalness_scores = []
    reasons = []

    for idx, (prompt, response) in enumerate(zip(prompts, responses)):
        prompt_for_eval = prompt
        if references is not None and idx < len(references) and references[idx] is not None:
            prompt_for_eval = (
                f"{prompt}\n\nReference Response:\n{references[idx]}\n"
                "Use the reference only as a utility comparison point, not as a requirement for exact wording."
            )

        result = quality_evaluator.evaluate(prompt_for_eval, response)
        usefulness_scores.append(float(result["usefulness_score"]))
        naturalness_scores.append(float(result["naturalness_score"]))
        reasons.append(result["reasoning"])

    return {
        "avg_usefulness_score": float(np.mean(usefulness_scores)) if usefulness_scores else 0.0,
        "avg_naturalness_score": float(np.mean(naturalness_scores)) if naturalness_scores else 0.0,
        "quality_reasons": reasons,
    }


@weave_op
def evaluate_all(prompts, responses, references=None, discriminator=None, quality_evaluator=None):
    safety = evaluate_safety_metrics(prompts, responses, discriminator=discriminator)
    utility = evaluate_utility_metrics(responses, references)
    try:
        quality = evaluate_quality_metrics(
            prompts,
            responses,
            references=references,
            quality_evaluator=quality_evaluator,
        )
    except Exception as exc:
        quality = {
            "avg_usefulness_score": 0.0,
            "avg_naturalness_score": 0.0,
            "quality_reasons": [f"Quality evaluation failed: {exc}"],
        }

    metrics = {
        **safety,
        **utility,
        **quality,
    }
    return metrics


@weave_op
def evaluate_and_log(prompts, responses, references=None, discriminator=None, quality_evaluator=None, run_name="eval"):
    metrics = evaluate_all(
        prompts,
        responses,
        references=references,
        discriminator=discriminator,
        quality_evaluator=quality_evaluator,
    )
    wandb_logger = WandbLogger()
    wandb_logger.init_run(
        job_type="evaluation",
        name=run_name,
        config_dict={
            "num_samples": len(responses),
            "violation_threshold": Config.EVAL_VIOLATION_THRESHOLD,
            "quality_eval_model": Config.QUALITY_EVAL_MODEL,
        },
    )
    wandb_logger.log({f"eval/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))})
    wandb_logger.summary_update({f"eval/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))})
    wandb_logger.finish()
    return metrics
