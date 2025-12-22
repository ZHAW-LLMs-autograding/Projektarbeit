import json
import os
from datetime import datetime

class Logger:
    """
    Schreibt Trainings-Logs immer in:
        metriken/logs/<run_name>/train_logs.jsonl
    """

    def __init__(self, run_name="default_run"):
        # zentraler Pfad
        base_dir = os.path.join("metriken", "logs", run_name)
        os.makedirs(base_dir, exist_ok=True)

        self.log_path = os.path.join(base_dir, "train_logs.jsonl")

    def log(self, data: dict):
        data["timestamp"] = datetime.now().isoformat()

        with open(self.log_path, "a") as f:
            f.write(json.dumps(data) + "\n")
