import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam
from typing import Optional, Dict, Any, Callable


class NeuralNetTrainer(pl.LightningModule):
    """
    A modular neural network trainer using PyTorch Lightning.
    
    This class can train any PyTorch model with configurable optimizer,
    loss function, and other training parameters.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer = Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        loss_fn: Optional[Callable] = None,
        learning_rate: float = 1e-3,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        scheduler_kwargs: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Callable]] = None
    ):
        """
        Initialize the neural network trainer.
        """
        super().__init__()
        
        self.model = model
        self.optimizer_class = optimizer
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.learning_rate = learning_rate
        self.scheduler_class = scheduler
        self.scheduler_kwargs = scheduler_kwargs or {}
        self.metrics = metrics or {}
        self.loss_fn = loss_fn
        self.save_hyperparameters(ignore=['model', 'loss_fn', 'metrics'])
    
    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        """Training step for PyTorch Lightning."""
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        
        # Log training loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Calculate and log additional metrics
        for metric_name, metric_fn in self.metrics.items():
            metric_value = metric_fn(y_hat, y)
            self.log(f'train_{metric_name}', metric_value, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step for PyTorch Lightning."""
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        
        # Log validation loss
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Calculate and log additional metrics
        for metric_name, metric_fn in self.metrics.items():
            metric_value = metric_fn(y_hat, y)
            self.log(f'val_{metric_name}', metric_value, on_step=False, on_epoch=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step for PyTorch Lightning."""
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        
        # Log test loss
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        
        # Calculate and log additional metrics
        for metric_name, metric_fn in self.metrics.items():
            metric_value = metric_fn(y_hat, y)
            self.log(f'test_{metric_name}', metric_value, on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler for PyTorch Lightning."""
        # Initialize optimizer
        optimizer = self.optimizer_class(
            self.parameters(), 
            lr=self.learning_rate, 
            **self.optimizer_kwargs
        )
        
        # Return just optimizer if no scheduler
        if self.scheduler_class is None:
            return optimizer
        
        # Initialize scheduler
        scheduler = self.scheduler_class(optimizer, **self.scheduler_kwargs)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
