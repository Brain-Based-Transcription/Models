from pytorch_lightning.callbacks import Callback
from torch.utils.data import TensorDataset, DataLoader
import torch


class MetricsLogger(Callback):
    """Simple callback to log training metrics"""
    def __init__(self, to_log: list[str]):
        self.to_log = to_log
        self.train_metrics = {metric: [] for metric in to_log}
        self.val_metrics = {metric: [] for metric in to_log}
    
    def on_train_epoch_end(self, trainer, pl_module):
        for metric in self.to_log:
            metric_value = trainer.callback_metrics.get(f'train_{metric}')
            if metric_value is not None:
                self.train_metrics[metric].append(metric_value.item())
    
    def on_validation_epoch_end(self, trainer, pl_module):
        for metric in self.to_log:
            metric_value = trainer.callback_metrics.get(f'val_{metric}')
            if metric_value is not None:
                self.val_metrics[metric].append(metric_value.item())


def create_data_loaders(X_train, Y_train, batch_size=32, val_split=0.2, loader_kwargs={}):
    """Create train and validation data loaders"""
    train_dataset = TensorDataset(X_train, Y_train)
    val_size = int(val_split * len(X_train))
    train_size = len(X_train) - val_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, **loader_kwargs)
    return train_loader, val_loader