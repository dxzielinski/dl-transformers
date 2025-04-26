import torch
import torch.nn as nn
import lightning as L
import torchmetrics


TASK = "multiclass"
NUM_CLASSES = 12


class _BNReLUConv(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, dropout
    ):
        super().__init__()
        self.add_module("bn", nn.BatchNorm2d(in_channels))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module(
            "conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
        )
        if dropout and dropout > 0:
            self.add_module("drop", nn.Dropout2d(p=dropout))


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, dropout):
        super().__init__()
        self.branch = _BNReLUConv(
            in_channels,
            growth_rate,
            kernel_size=3,
            stride=1,
            padding=1,
            dropout=dropout,
        )

    def forward(self, x):
        out = self.branch(x)
        return torch.cat([x, out], dim=1)


class Transition(nn.Sequential):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        self.add_module("bn", nn.BatchNorm2d(in_channels))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module(
            "conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
        )
        if dropout and dropout > 0:
            self.add_module("drop", nn.Dropout2d(p=dropout))
        self.add_module("pool", nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    def __init__(
        self,
        depth=40,
        first_output=16,
        growth_rate=12,
        dropout=0.2,
        num_classes=NUM_CLASSES,
    ):
        super().__init__()
        assert (depth - 4) % 3 == 0, "depth must be 3n+4"
        num_layers_per_block = (depth - 4) // 3

        self.conv1 = nn.Conv2d(
            1, first_output, kernel_size=3, stride=1, padding=1, bias=False
        )
        num_channels = first_output

        self.block1 = self._make_dense_block(
            num_layers_per_block, num_channels, growth_rate, dropout
        )
        num_channels += num_layers_per_block * growth_rate
        self.trans1 = Transition(num_channels, num_channels, dropout)

        self.block2 = self._make_dense_block(
            num_layers_per_block, num_channels, growth_rate, dropout
        )
        num_channels += num_layers_per_block * growth_rate
        self.trans2 = Transition(num_channels, num_channels, dropout)

        self.block3 = self._make_dense_block(
            num_layers_per_block, num_channels, growth_rate, dropout
        )
        num_channels += num_layers_per_block * growth_rate

        self.bn = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_channels, num_classes)

    def _make_dense_block(self, num_layers, in_channels, growth_rate, dropout):
        layers = []
        for i in range(num_layers):
            layers.append(
                DenseLayer(in_channels + i * growth_rate, growth_rate, dropout)
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.trans1(x)
        x = self.block2(x)
        x = self.trans2(x)
        x = self.block3(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        return logits


class DenseNetLit(L.LightningModule):
    def __init__(self, hyperparameters):
        """
        Example hyperparameters:
        depth=40, first_output=16, growth_rate=12, dropout=0.2
        hyperparameters = {
            "depth": 40,
            "first_output": 16,
            "growth_rate": 12,
            "dropout": 0.2,
        }
        """
        super().__init__()
        self.model = DenseNet(
            hyperparameters["depth"],
            hyperparameters["first_output"],
            hyperparameters["growth_rate"],
            hyperparameters["dropout"],
            num_classes=NUM_CLASSES,
        )
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
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=0.1, weight_decay=10e-6, momentum=0.9
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=4.0,
            steps_per_epoch=67,
            epochs=150,
            div_factor=40,
            final_div_factor=40,
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
