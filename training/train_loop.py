import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models import BaseModule


__all__ = [
    "TrainLoop",
]


class TrainLoop:
    def __init__(
            self,
            model: BaseModule,
            optimizer: Optimizer,
            loss_fn: nn.Module,
            train_data: Dataset,
            epoches: int,
            batch_size: int = 64,
            test_every=5,
            val_data: Dataset = None,
            save_every: int = 0,
            save_file: str = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_data = train_data
        self.epoches = epoches
        self.batch_size = batch_size
        self.test_every = test_every
        self.val_data = val_data
        self.save_every = save_every
        self.save_file = save_file

        self.trained_epoches = 0
        self.train_acc = []
        self.train_loss = []
        self.train_epoches = []
        self.val_acc = []
        self.val_loss = []
        self.val_epoches = []

    def _train_one_epoch(self):
        loss = 0
        correct = 0
        total = 0
        batches = 0

        self.model.train()
        train_loader = DataLoader(self.train_data, shuffle=True, batch_size=self.batch_size)
        mini_batches = (len(self.train_data) + self.batch_size - 1) // self.batch_size

        with tqdm(total=mini_batches) as prog:
            for X_train, y_train in train_loader:
                logits = self.model(X_train)
                batch_loss = self.loss_fn(logits, y_train)

                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

                class_pred = torch.argmax(logits, dim=-1)
                class_train = torch.argmax(y_train, dim=-1)

                correct += torch.sum(class_pred == class_train).item()
                total += len(class_train)
                loss += batch_loss
                batches += 1
                prog.update(1)

        loss /= batches
        acc = correct / total

        self.train_acc.append(acc)
        self.train_loss.append(loss)

    def _validate(self):
        loss = 0
        correct = 0
        total = 0
        batches = 0

        val_loader = DataLoader(self.val_data, batch_size=len(self.val_data))
        for X_val, y_val in val_loader:
            logits = self.model.infer(X_val)
            batch_loss = self.loss_fn(logits, y_val)

            class_pred = torch.argmax(logits, dim=-1)
            class_val = torch.argmax(y_val, dim=-1)

            correct += torch.sum(class_pred == class_val).item()
            total += len(class_val)
            loss += batch_loss
            batches += 1

        loss /= batches
        acc = correct / total

        self.val_acc.append(acc)
        self.val_loss.append(loss)

    def train(self):
        for epoch in range(self.trained_epoches + 1, self.trained_epoches + 1 + self.epoches):
            print("-" * 20 + " " + f"epoch {epoch}" + " " + "-" * 20)
            self._train_one_epoch()
            print(f"train loss: {self.train_loss[-1]: .6f}")
            print(f"train accuracy: {self.train_acc[-1]: .2%}")
            print("-" * 51)

            if (epoch == 1 or epoch % self.test_every == 0) and self.val_data and self.test_every:
                print("-" * 12 + " " + f"running test at epoch {epoch}" + " " + "-" * 12)
                self._validate()
                print(f"validation loss: {self.val_loss[-1]: .6f}")
                print(f"validation accuracy: {self.val_acc[-1]: .2%}")
                print("-" * 51)
