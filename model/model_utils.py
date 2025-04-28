"""
Utility functions and classes for model training.
"""

import timm
import torch
import lightning as L
import torchmetrics
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchtune.training import get_cosine_schedule_with_warmup

from model_config import TASK, NUM_CLASSES
from spectrogram_transformer import SpectrogramTransformer


class GreyscaleTransformer(torch.nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 2,
        d_model: int = 128,
        nhead: int = 2,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 1,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        assert image_size % patch_size == 0, (
            "Image dimensions must be divisible by the patch size"
        )
        num_patches = (image_size // patch_size) ** 2
        self.patch_embed = torch.nn.Conv2d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, num_patches + 1, d_model))

        self.transformer = torch.nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
        )

        self.norm = torch.nn.LayerNorm(d_model)
        self.fc = torch.nn.Linear(d_model, num_classes)

        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        torch.nn.init.trunc_normal_(self.patch_embed.weight, std=0.02)
        if self.patch_embed.bias is not None:
            torch.nn.init.zeros_(self.patch_embed.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, _, _ = x.shape
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        src = x[:, 1:, :].permute(1, 0, 2)
        tgt = x[:, :1, :].permute(1, 0, 2)
        out = self.transformer(src=src, tgt=tgt)
        out = out.squeeze(0)
        out = self.norm(out)
        logits = self.fc(out)
        return logits


def create_vit_tiny_greyscale(
    num_classes: int = 12, pretrained: bool = False
) -> torch.nn.Module:
    """
    Creates a ViT-Tiny model adapted for:
      - 1-channel (greyscale) input
      - custom number of classes
    """
    model = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=pretrained,
        in_chans=1,
        num_classes=num_classes,
    )

    return model


def create_deit_tiny_greyscale(
    num_classes: int = 12, pretrained: bool = False
) -> torch.nn.Module:
    """
    Creates a DeiT-Tiny (distilled) model adapted for:
      - 1-channel (greyscale) input
      - custom number of classes
    """
    model = timm.create_model(
        "deit_tiny_distilled_patch16_224",
        pretrained=pretrained,
        in_chans=1,
        num_classes=num_classes,
    )
    return model


class ModelTransformer(L.LightningModule):
    def __init__(
        self,
        hyperparameters,
        weight=None,
        model=SpectrogramTransformer(num_classes=NUM_CLASSES, patch_size=4),
    ):
        """
        Args:
            hyperparameters: dict containing model hyperparameters - it must contain the
                following keys:
                - learning_rate: float, learning rate for the optimizer
                - dropout: float, dropout rate before the last linear layer
                - weight_decay: float, weight decay for the optimizer
        """
        super().__init__()
        self.model = model
        self.learning_rate = hyperparameters["learning_rate"]
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
        self.train_metrics = torchmetrics.MetricCollection(
            {
                "f1_macro": torchmetrics.F1Score(
                    task=TASK, num_classes=NUM_CLASSES, average="macro"
                ),
                "precision": torchmetrics.Precision(
                    task=TASK, num_classes=NUM_CLASSES, average="macro"
                ),
                "recall": torchmetrics.Recall(
                    task=TASK, num_classes=NUM_CLASSES, average="macro"
                ),
                "auroc": torchmetrics.AUROC(
                    task=TASK, num_classes=NUM_CLASSES, average="macro"
                ),
                "accuracy": torchmetrics.Accuracy(task=TASK, num_classes=NUM_CLASSES),
            },
            prefix="train_",
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")
        self.train_batch_outputs = []
        self.val_batch_outputs = []
        self.test_batch_outputs = []
        self.hyperparameters = hyperparameters

    def on_train_start(self):
        if self.logger is not None:
            self.logger.log_hyperparams(self.hyperparameters)

    def on_test_start(self):
        if self.logger is not None:
            self.logger.log_hyperparams(self.hyperparameters)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        probabilities = torch.softmax(logits, dim=1)
        self.train_batch_outputs.append({"probabilities": probabilities, "y": y})
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log(
            "lr", self.optimizers().param_groups[0]["lr"], on_step=True, on_epoch=True
        )
        return loss

    def on_train_epoch_end(self):
        probabilities = torch.cat(
            [x["probabilities"] for x in self.train_batch_outputs]
        )
        y = torch.cat([x["y"] for x in self.train_batch_outputs])
        metrics = self.train_metrics(probabilities, y)
        self.log_dict(metrics)
        self.train_metrics.reset()
        self.train_batch_outputs.clear()

    def validation_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        probabilities = torch.softmax(logits, dim=1)
        self.val_batch_outputs.append({"probabilities": probabilities, "y": y})
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        probabilities = torch.cat([x["probabilities"] for x in self.val_batch_outputs])
        y = torch.cat([x["y"] for x in self.val_batch_outputs])
        metrics = self.val_metrics(probabilities, y)
        self.log_dict(metrics)
        self.val_metrics.reset()
        self.val_batch_outputs.clear()

    def test_step(self, batch):
        x, y = batch
        logits = self(x)
        probabilities = torch.softmax(logits, dim=1)
        self.test_batch_outputs.append({"probabilities": probabilities, "y": y})

    def on_test_epoch_end(self):
        probabilities = torch.cat([x["probabilities"] for x in self.test_batch_outputs])
        y = torch.cat([x["y"] for x in self.test_batch_outputs])
        metrics = self.test_metrics(probabilities, y)
        self.log_dict(metrics)
        self.test_metrics.reset()
        self.test_batch_outputs.clear()

    def configure_optimizers(self):
        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name for nd in ["bias", "LayerNorm.weight"]):
                no_decay.append(param)
            else:
                decay.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.hyperparameters["weight_decay"]},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.hyperparameters["learning_rate"],
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(0.1 * total_steps)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            num_cycles=0.5,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "strict": False,
            },
        }


##### Data module for classification task #####
IMAGE_SIZE = 32
mean, std = [0.4914], [0.247]

transforms_train = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)


transforms_test = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)


class ClassificationData(L.LightningDataModule):
    def __init__(
        self,
        data_dir="../data-no-noise-no-silence",
        batch_size=512,
        transforms_train=transforms_train,
        transforms_test=transforms_test,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transforms_train = transforms_train
        self.transforms_test = transforms_test

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == "fit":
            self.train_dataset = ImageFolder(
                root=f"{self.data_dir}/train", transform=self.transforms_train
            )
            self.val_dataset = ImageFolder(
                root=f"{self.data_dir}/val", transform=self.transforms_test
            )
        if stage == "test":
            self.test_dataset = ImageFolder(
                root=f"{self.data_dir}/test", transform=self.transforms_test
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=31,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=31,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=31,
            persistent_workers=True,
        )
