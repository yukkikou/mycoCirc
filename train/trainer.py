"""
Training loop utilities for PanCirc-Fungi.
"""

import logging
import os
import time
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.logging import AverageMeter

logger = logging.getLogger(__name__)


class Trainer:
    """Generic trainer for PyTorch models.

    Parameters
    ----------
    model : nn.Module
    config : dict
        Training configuration (learning_rate, weight_decay, etc.).
    device : str
        "cuda" or "cpu".
    """

    def __init__(self, model: nn.Module, config: dict,
                 device: str = "cuda", task: str = "pretrain"):
        self.model = model.to(device)
        self.task = task
        self.config = config
        self.device = device
        # Safely coerce numeric config values (YAML may parse "1e-5" as string)
        lr = float(config.get("learning_rate", 1e-3))
        wd = float(config.get("weight_decay", 1e-5))
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=wd,
        )
        self.scheduler = None
        self.scaler = torch.amp.GradScaler() if device == "cuda" else None

        self.start_epoch = 0
        self.best_val_loss = float("inf")
        self.train_metrics: Dict = {}
        self.val_metrics: Dict = {}

    def train_epoch(self, loader: DataLoader,
                    loss_fn: Callable) -> Dict[str, float]:
        """Run one training epoch.

        Args:
            loader: training DataLoader
            loss_fn: callable(outputs, batch) -> dict of losses
        Returns:
            dict of average losses
        """
        self.model.train()
        meters = {}

        for batch_idx, batch in enumerate(loader):
            batch = self._to_device(batch)

            # Forward
            with torch.amp.autocast(device_type=self.device,
                                    enabled=self.scaler is not None):
                outputs = self.model(batch, task=self.task)
                losses = loss_fn(outputs, batch)
                total_loss = sum(losses.values())

            # Backward
            self.optimizer.zero_grad()
            if self.scaler:
                self.scaler.scale(total_loss).backward()
                if self.config.get("gradient_clip", 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config["gradient_clip"],
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                if self.config.get("gradient_clip", 0) > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config["gradient_clip"],
                    )
                self.optimizer.step()

            # Track metrics
            for k, v in losses.items():
                if k not in meters:
                    meters[k] = AverageMeter()
                meters[k].update(v.item(), batch["is_positive"].size(0))

            if batch_idx % self.config.get("log_interval", 50) == 0:
                log = [f"{k}: {m.avg:.4f}" for k, m in meters.items()]
                logger.info(f"  Train batch {batch_idx}: {', '.join(log)}")

        if self.scheduler:
            self.scheduler.step()

        return {k: m.avg for k, m in meters.items()}

    @torch.no_grad()
    def validate(self, loader: DataLoader,
                 loss_fn: Callable) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        meters = {}

        for batch in loader:
            batch = self._to_device(batch)
            outputs = self.model(batch, task=self.task)
            losses = loss_fn(outputs, batch)

            for k, v in losses.items():
                if k not in meters:
                    meters[k] = AverageMeter()
                meters[k].update(v.item(), batch["is_positive"].size(0))

        return {k: m.avg for k, m in meters.items()}

    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            n_epochs: int, loss_fn: Callable,
            checkpoint_dir: str = "checkpoints",
            early_stop_patience: Optional[int] = None):
        """Full training loop."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        patience_counter = 0
        best_epoch = 0

        for epoch in range(self.start_epoch, self.start_epoch + n_epochs):
            logger.info(f"Epoch {epoch+1}/{self.start_epoch + n_epochs}")
            t0 = time.time()

            train_losses = self.train_epoch(train_loader, loss_fn)
            val_losses = self.validate(val_loader, loss_fn)

            epoch_time = time.time() - t0
            val_total = sum(val_losses.values())

            logger.info(
                f"  [{epoch+1}] "
                f"Train loss: {sum(train_losses.values()):.4f} "
                f"Val loss: {val_total:.4f} "
                f"({epoch_time:.1f}s)"
            )

            # Checkpoint
            if val_total < self.best_val_loss:
                self.best_val_loss = val_total
                best_epoch = epoch
                patience_counter = 0
                ckpt_path = os.path.join(checkpoint_dir, "best.pt")
                self.save_checkpoint(ckpt_path, epoch)
                logger.info(f"  ✓ New best model saved to {ckpt_path}")
            else:
                patience_counter += 1

            # Early stopping
            if (early_stop_patience
                    and patience_counter >= early_stop_patience):
                logger.info(
                    f"  Early stopping at epoch {epoch+1} "
                    f"(best was epoch {best_epoch+1})"
                )
                break

            # Periodic save
            if (epoch + 1) % self.config.get("save_interval", 5) == 0:
                ckpt_path = os.path.join(
                    checkpoint_dir, f"epoch_{epoch+1}.pt"
                )
                self.save_checkpoint(ckpt_path, epoch)

    def save_checkpoint(self, path: str, epoch: int):
        """Save model checkpoint."""
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.start_epoch = ckpt.get("epoch", 0) + 1
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        logger.info(f"Loaded checkpoint from {path} (epoch {ckpt.get('epoch', 0)+1})")

    def _to_device(self, batch: Dict) -> Dict:
        """Move batch tensors to device."""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
