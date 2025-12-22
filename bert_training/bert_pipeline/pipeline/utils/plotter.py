import json
import os
import matplotlib.pyplot as plt

class Plotter:
    """
    Plotter für Trainingsmetriken und Evaluationsplots.
    Speichert automatisch nach:
        metriken/plots/<run_name>/
    """

    def __init__(self, run_name="default_run"):
        # Logs befinden sich hier:
        self.log_dir = os.path.join("metriken", "logs", run_name)

        # Plots sollen hier hin:
        self.output_dir = os.path.join("metriken", "plots", run_name)
        os.makedirs(self.output_dir, exist_ok=True)

    # ---------------------------------------------------------------
    # Training Loss Plot
    # ---------------------------------------------------------------
    def plot_training_loss(self):
        log_file = os.path.join(self.log_dir, "train_logs.jsonl")

        if not os.path.exists(log_file):
            print(f"[WARN] Keine Logs gefunden unter {log_file}")
            return

        steps = []
        losses = []

        with open(log_file) as f:
            for line in f:
                entry = json.loads(line)
                steps.append(entry["step"])
                losses.append(entry["train_loss"])

        if len(steps) == 0:
            print("[WARN] Log-Datei enthält keine Daten.")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(steps, losses, marker='.')
        plt.xlabel("Step")
        plt.ylabel("Train Loss")
        plt.title("Training Loss Curve")
        plt.grid(True)

        out_path = os.path.join(self.output_dir, "training_loss.png")
        plt.savefig(out_path)
        plt.close()

        print(f"[OK] Training-Loss-Plot gespeichert unter: {out_path}")
