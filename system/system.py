import pytorch_lightning as pl
from loss.loss import get_loss
from optimizer.optimizer import get_optimizer
from scheduler.scheduler import get_scheduler

import torch
import numpy as np
from pytorch_lightning.metrics import Accuracy


class LitClassifier(pl.LightningModule):
    def __init__(self, hparams, model):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = model
        self.criteria = get_loss(hparams.training.loss)
        self.accuracy = Accuracy()

    def forward(self, x):
        # use forward for inference/predictions
        return self.model(x)

    def configure_optimizers(self):
        optimizer = get_optimizer(self.model.parameters(), self.hparams.training.optimizer)

        scheduler = get_scheduler(optimizer, self.hparams.training.scheduler)
        
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criteria(y_hat, y)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criteria(y_hat, y)
        
        return {
            "val_loss": loss,
            "y": y,
            "y_hat": y_hat
            }
    
    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        y = torch.cat([x["y"] for x in outputs]).cpu()
        y_hat = torch.cat([x["y_hat"] for x in outputs]).cpu()

        preds = np.argmax(y_hat, axis=1)

        val_accuracy = self.accuracy(y, preds)

        self.log('avg_val_loss', avg_val_loss)
        self.log('val_acc', val_accuracy)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criteria(y_hat, y)
        self.log('test_loss', loss)

    