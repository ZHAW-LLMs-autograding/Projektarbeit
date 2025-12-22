from .base_trainer import BaseTrainer

class SentenceBertTrainer(BaseTrainer):
    def train_epoch(self):
        train_dataloader = self.dm.dataloader("train")
        loss_fn = self.model.loss_fn
        total_loss = 0
        for batch in train_dataloader:
            loss_value = loss_fn(self.model.model, batch)
            loss_value.backward()
            total_loss += loss_value.item()
        return total_loss / len(train_dataloader)

    def evaluate(self):
        val_loader = self.dm.dataloader("validation")
        total_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                loss_value = self.model.loss_fn(self.model.model, batch)
                total_loss += loss_value.item()
        return total_loss / len(val_loader)
