# main.py
import random

from agent import PPOAgent
from config import Config
from dataset_utils import load_local_hh_dataset, load_local_pku_dataset
from discriminator import SafetyDiscriminator
from evaluation import evaluate_and_log
from models import LanguageModelPolicy
from openai_evaluator import OpenAIHumanReviewer, OpenAISafetyEvaluator
from scheduler import FeedbackScheduler
from tqdm.auto import tqdm
from utils import get_human_input, set_seed
from wandb_logger import WandbLogger
from weave_support import init_weave


def get_hh_prompts(num_samples=10):
    """
    Load prompts from Anthropic/hh-rlhf by keeping the final `Assistant:`
    marker so the policy can continue the dialogue.
    """
    print("Loading Anthropic/hh-rlhf dataset from local cache...")
    raw = load_local_hh_dataset()
    dataset = raw["train"].select(range(min(num_samples, len(raw["train"]))))

    prompts = []
    for item in dataset:
        full_text = item["chosen"]
        split_token = "Assistant:"
        if split_token in full_text:
            prompt_body, reference = full_text.rsplit(split_token, 1)
            prompt = prompt_body + split_token
            prompts.append({"source": "HH", "prompt": prompt, "reference": reference.strip()})
        else:
            prompts.append({"source": "HH", "prompt": full_text[:50], "reference": full_text})

    return prompts


def _format_pku_prompt(prompt_text: str) -> str:
    cleaned = (prompt_text or "").strip()
    return f"\n\nHuman: {cleaned}\n\nAssistant:"


def _pick_pku_reference(row) -> str:
    response_0 = (row.get("response_0") or "").strip()
    response_1 = (row.get("response_1") or "").strip()
    response_0_safe = bool(row.get("is_response_0_safe", True))
    response_1_safe = bool(row.get("is_response_1_safe", True))

    if response_0_safe and response_0:
        return response_0
    if response_1_safe and response_1:
        return response_1
    return response_0 or response_1 or ""


def get_pku_prompts(num_samples=3):
    print("Loading PKU-SafeRLHF prompts from local cache...")
    raw = load_local_pku_dataset()
    dataset = raw["train"]

    unsafe_rows = []
    safe_rows = []
    seen_prompts = set()

    for row in dataset:
        prompt = (row.get("prompt") or "").strip()
        if not prompt or prompt in seen_prompts:
            continue
        seen_prompts.add(prompt)

        is_unsafe = (not bool(row.get("is_response_0_safe", True))) or (not bool(row.get("is_response_1_safe", True)))
        item = {
            "source": "PKU",
            "prompt": _format_pku_prompt(prompt),
            "reference": _pick_pku_reference(row),
        }
        if is_unsafe:
            unsafe_rows.append(item)
        else:
            safe_rows.append(item)

        if len(unsafe_rows) >= max(num_samples * 4, 50) and len(safe_rows) >= max(num_samples * 2, 20):
            break

    rng = random.Random(Config.SEED)
    rng.shuffle(unsafe_rows)
    rng.shuffle(safe_rows)

    unsafe_target = min(len(unsafe_rows), max(1, int(round(num_samples * Config.PKU_UNSAFE_PROMPT_RATIO))))
    safe_target = min(len(safe_rows), max(0, num_samples - unsafe_target))

    selected = unsafe_rows[:unsafe_target] + safe_rows[:safe_target]
    if len(selected) < num_samples:
        remainder = unsafe_rows[unsafe_target:] + safe_rows[safe_target:]
        selected.extend(remainder[: num_samples - len(selected)])

    rng.shuffle(selected)
    return selected


def get_training_prompts(num_samples: int):
    pku_count = min(num_samples, max(1, int(round(num_samples * Config.PKU_PROMPT_MIX_RATIO))))
    hh_count = max(0, num_samples - pku_count)

    hh_prompts = get_hh_prompts(num_samples=hh_count)
    pku_prompts = get_pku_prompts(num_samples=pku_count)

    prompts = hh_prompts + pku_prompts
    rng = random.Random(Config.SEED)
    rng.shuffle(prompts)
    return prompts


def main():
    set_seed(Config.SEED)
    init_weave()
    print("Initializing Hybrid Alignment System...")

    prompts = get_training_prompts(num_samples=Config.NUM_PROMPTS)
    total_steps = Config.EPOCHS * len(prompts)
    wandb_logger = WandbLogger()
    wandb_logger.init_run(
        job_type="policy_train",
        name="main-train",
        config_dict={
            "model_name": Config.MODEL_NAME,
            "epochs": Config.EPOCHS,
            "max_gen_len": Config.MAX_GEN_LEN,
            "learning_rate": Config.LEARNING_RATE,
            "risk_threshold_low": Config.RISK_THRESHOLD_LOW,
            "risk_threshold_high": Config.RISK_THRESHOLD_HIGH,
            "uncertainty_threshold": Config.UNCERTAINTY_THRESHOLD,
            "human_budget_percent": Config.HUMAN_BUDGET_PERCENT,
            "num_prompts": Config.NUM_PROMPTS,
            "pku_prompt_mix_ratio": Config.PKU_PROMPT_MIX_RATIO,
            "pku_unsafe_prompt_ratio": Config.PKU_UNSAFE_PROMPT_RATIO,
            "openai_eval_model": Config.OPENAI_EVAL_MODEL,
            "human_eval_model": Config.HUMAN_EVAL_MODEL,
            "total_steps": total_steps,
        },
    )

    policy = LanguageModelPolicy()
    discriminator = SafetyDiscriminator()
    ai_evaluator = OpenAISafetyEvaluator()
    human_reviewer = OpenAIHumanReviewer()
    scheduler = FeedbackScheduler(total_steps=total_steps)
    agent = PPOAgent(policy)

    hh_prompt_count = sum(1 for item in prompts if item["source"] == "HH")
    pku_prompt_count = sum(1 for item in prompts if item["source"] == "PKU")
    print(f"\nStart Training Loop for {len(prompts)} episodes using mixed HH/PKU prompts...\n")
    print(f"Prompt mix: HH={hh_prompt_count}, PKU={pku_prompt_count}")
    print(
        f"Human review budget: {scheduler.human_budget}/{total_steps} "
        f"({scheduler.human_budget_percent:.0%})"
    )
    print("-" * 50)

    global_step = 0
    eval_prompts = []
    eval_references = []
    eval_responses = []
    progress_bar = tqdm(total=total_steps, desc="Training", dynamic_ncols=True)
    for epoch in range(Config.EPOCHS):
        for i, prompt_item in enumerate(prompts):
            prompt_source = prompt_item["source"]
            prompt_text = prompt_item["prompt"]
            prompt_reference = prompt_item.get("reference", "")
            global_step += 1
            response_ids, response_text, inputs_ids = policy.generate(prompt_text)

            class_probs, predicted_cls, uncertainty = discriminator.predict(response_text)
            local_risk_score = discriminator.risk_score(response_text)
            top_risks = discriminator.top_risks(class_probs, k=3)

            source_type, need_human = scheduler.decide(local_risk_score, uncertainty)

            human_label = None
            ai_reasoning = None
            ai_risk_score = local_risk_score
            human_reasoning = None
            human_risk_score = None

            if need_human:
                try:
                    human_result = human_reviewer.evaluate(prompt_text, response_text)
                    human_label = int(human_result["label"])
                    human_risk_score = float(human_result["risk_score"])
                    human_reasoning = human_result["reasoning"]
                    human_risk_multihot = human_result["risk_multihot"]
                except Exception as exc:
                    print(f"  [Human Sim] Fallback via utils.get_human_input(): {exc}")
                    human_result = get_human_input(prompt_text, response_text)
                    human_label = int(human_result["label"])
                    human_risk_score = float(human_result["risk_score"])
                    human_reasoning = human_result["reasoning"]
                    human_risk_multihot = human_result["risk_multihot"]

                disc_loss = discriminator.update_with_replay(
                    response_text,
                    human_risk_multihot if human_label == -1 else human_label,
                    batch_size=8,
                )
                print(f"  [Discriminator Update] Feedback received. Replay Loss: {disc_loss:.4f}")
            else:
                try:
                    eval_result = ai_evaluator.evaluate(prompt_text, response_text)
                    ai_risk_score = float(eval_result["risk_score"])
                    ai_reasoning = eval_result["reasoning"]
                except Exception as exc:
                    print(f"  [OpenAI Eval] Fallback to local discriminator score: {exc}")

            raw_reward = scheduler.get_reward_source(source_type, ai_risk_score, human_label)
            loss, kl = agent.update(inputs_ids, response_ids, raw_reward)

            print(
                f"Epoch {epoch} | Step {i} | Source: {prompt_source} | "
                f"Risk: {local_risk_score:.2f} | U: {uncertainty:.2f} | "
                f"Reward: {raw_reward:.2f} | Loss: {loss:.4f}"
            )
            print("-" * 30)
            progress_bar.update(1)
            progress_bar.set_postfix(
                risk=f"{local_risk_score:.2f}",
                u=f"{uncertainty:.2f}",
                reward=f"{raw_reward:.2f}",
                loss=f"{loss:.4f}",
            )

            if epoch == Config.EPOCHS - 1:
                eval_prompts.append(prompt_text)
                eval_references.append(prompt_reference)
                eval_responses.append(response_text)

            wandb_logger.log(
                {
                    "train/epoch": epoch,
                    "train/step_in_epoch": i,
                    "train/global_step": global_step,
                    "train/prompt_is_pku": int(prompt_source == "PKU"),
                    "train/local_risk_score": local_risk_score,
                    "train/uncertainty": uncertainty,
                    "train/ai_eval_risk": ai_risk_score,
                    "train/raw_reward": raw_reward,
                    "train/policy_loss": loss,
                    "train/kl": kl,
                    "train/human_budget_remaining": scheduler.human_budget,
                    "train/used_human_review": int(need_human),
                },
                step=global_step,
            )

    wandb_logger.summary_update(
        {
            "summary/human_budget_remaining": scheduler.human_budget,
            "summary/total_steps": total_steps,
        }
    )
    progress_bar.close()

    print("\nRunning final evaluation...")
    final_metrics = evaluate_and_log(
        eval_prompts,
        eval_responses,
        references=eval_references,
        discriminator=discriminator,
        run_name="final-eval",
    )
    print("Final evaluation metrics:")
    for key, value in final_metrics.items():
        if key == "quality_reasons":
            continue
        print(f"  {key}: {value}")

    wandb_logger.finish()


if __name__ == "__main__":
    main()
