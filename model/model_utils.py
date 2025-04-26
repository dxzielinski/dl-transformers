"""
Utility functions and classes for model training.

Main focus: implementing dynamic transformer model architecture.

Default, fixed hyperparameters are:

- optimizer: AdamW
- loss function: CrossEntropyLoss
- learning rate schedule: Noam or linear warmup with cosine decay or ReduceLROnPlateau
- learning rate - it will be found by learning rate tuning
- activation function: RELU

Hyperparameter space for the model includes:

- number of layers [4 to 10]
- number of attention heads [4 to 12]
- hidden size [256 to 768]
- dropout rate [0.05 to 0.15]
- weight decay [0 to 0.01]
"""

import torch
import lightning as L
import torchmetrics
from torchvision.datasets import DatasetFolder


from model_config import TASK, NUM_CLASSES


def _calc_conv_output_size(
    shape=(80, 32), kernel_size=2, stride=1, padding=0, pooling_size=2
):
    height, width = shape
    height_conv_out = (height - kernel_size + 2 * padding) // stride + 1
    height_pool_out = height_conv_out // pooling_size
    width_conv_out = (width - kernel_size + 2 * padding) // stride + 1
    width_pool_out = width_conv_out // pooling_size
    return height_pool_out, width_pool_out


class BasicCNN(torch.nn.Module):
    def __init__(self, dropout):
        super().__init__()
        h1, w1 = _calc_conv_output_size(pooling_size=1)
        h2, w2 = _calc_conv_output_size((h1, w1), pooling_size=1)
        h3, w3 = _calc_conv_output_size((h2, w2), pooling_size=1)
        flatten_dim = 128 * h3 * w3

        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=2, stride=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(1),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(1),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(
                in_features=flatten_dim,
                out_features=64,
            ),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(in_features=64, out_features=NUM_CLASSES),
        )

    def forward(self, x):
        return self.layers(x)


class Model(L.LightningModule):
    def __init__(self, hyperparameters):
        """
        Args:
            hyperparameters: dict containing model hyperparameters - it must contain the
                following keys:
                - learning_rate: float, learning rate for the optimizer
                - dropout: float, dropout rate before the last linear layer
                - weight_decay: float, weight decay for the optimizer
        """
        super().__init__()
        self.model = BasicCNN(hyperparameters["dropout"])
        self.learning_rate = hyperparameters["learning_rate"]
        self.loss_fn = torch.nn.CrossEntropyLoss()
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
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.hyperparameters["weight_decay"],
            momentum=0.9,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer, max_lr=4.0, epochs=50, steps_per_epoch=177
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


def pt_loader(path):
    return torch.load(path)


class ClassificationData(L.LightningDataModule):
    def __init__(self, data_dir="../data", batch_size=16, transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transform

    def setup(self, stage=None):
        if stage == "fit":
            self.train_dataset = DatasetFolder(
                root=f"{self.data_dir}/train",
                transform=self.transform,
                loader=pt_loader,
                extensions=[".pt"],
            )
            self.val_dataset = DatasetFolder(
                root=f"{self.data_dir}/val",
                transform=self.transform,
                loader=pt_loader,
                extensions=[".pt"],
            )
        if stage == "test":
            self.test_dataset = DatasetFolder(
                root=f"{self.data_dir}/test",
                transform=self.transform,
                loader=pt_loader,
                extensions=[".pt"],
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=31,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=31,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=31,
        )
