from config import Config


class WandbLogger:
    def __init__(self):
        self.enabled = False
        self.run = None
        self._wandb = None

    def init_run(self, *, job_type: str, config_dict: dict, name: str = ""):
        if not Config.WANDB_ENABLED:
            return

        try:
            import wandb
        except Exception as exc:
            print(f"[W&B] wandb import failed, logging disabled: {exc}")
            return

        init_kwargs = {
            "project": Config.WANDB_PROJECT,
            "config": config_dict,
            "job_type": job_type,
            "mode": Config.WANDB_MODE,
            "reinit": True,
        }
        if Config.WANDB_ENTITY:
            init_kwargs["entity"] = Config.WANDB_ENTITY
        run_name = name or Config.WANDB_RUN_NAME
        if run_name:
            init_kwargs["name"] = run_name

        try:
            self.run = wandb.init(**init_kwargs)
            self._wandb = wandb
            self.enabled = True
        except Exception as exc:
            print(f"[W&B] init failed, logging disabled: {exc}")

    def log(self, metrics: dict, step: int | None = None):
        if not self.enabled or self.run is None:
            return
        try:
            if step is None:
                self.run.log(metrics)
            else:
                self.run.log(metrics, step=step)
        except Exception as exc:
            print(f"[W&B] log failed: {exc}")

    def summary_update(self, metrics: dict):
        if not self.enabled or self.run is None:
            return
        try:
            for key, value in metrics.items():
                self.run.summary[key] = value
        except Exception as exc:
            print(f"[W&B] summary update failed: {exc}")

    def finish(self):
        if not self.enabled or self.run is None:
            return
        try:
            self.run.finish()
        except Exception as exc:
            print(f"[W&B] finish failed: {exc}")
        finally:
            self.run = None
            self.enabled = False
