import torch
from torch.optim import AdamW
from .base_trainer import BaseTrainer

class MixtralTrainer(BaseTrainer):
    def train_epoch(self):
        self.model.model.train()
        dataloader = self.dm.dataloader("train")
        optimizer = AdamW(self.model.model.parameters(), lr=self.cfg.get("lr", 1e-5))
        total_loss = 0
        for batch in dataloader:
            outputs = self.model.forward(batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def evaluate(self):
        self.model.model.eval()
        dataloader = self.dm.dataloader("validation")
        total_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                outputs = self.model.forward(batch)
                total_loss += outputs.loss.item()
        return total_loss / len(dataloader)
