from .llama_trainer import LlamaTrainer

class TrainerFactory:
    @staticmethod
    def create(trainer_type: str, model_adapter, datamodule, cfg, logger=None):
        trainer_type = trainer_type.lower()

        if trainer_type in ["llama", "llama_regression", "llama3.1"]:
            # ⬇ richtige Argumentreihenfolge!
            return LlamaTrainer(cfg, model_adapter, datamodule, logger)
        else:
            raise ValueError(f"Unknown trainer type: {trainer_type}")
