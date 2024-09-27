import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from tqdm import tqdm
from pathlib import Path
import math

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
            train_dataloader: DataLoader,
            epoches: int,
            device: str,
            lr_scheduler: LRScheduler = None,
            test_every: int = 5,
            val_dataloader: DataLoader = None,
            save_every: int = 0,
            root: str = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_dataloader = train_dataloader
        self.epoches = epoches
        self.device = device
        self.lr_scheduler = lr_scheduler
        self.test_every = test_every
        self.val_dataloader = val_dataloader
        self.save_every = save_every
        self.root = Path(root).resolve()

        self.train_acc = []
        self.train_loss = []
        self.train_epoches = []
        self.val_acc = []
        self.val_loss = []
        self.val_epoches = []

    def _train_one_epoch(self):
        loss = torch.tensor(0.0).to(self.device)
        correct = torch.tensor(0.0).to(self.device)

        self.model.train()
        dataset_len = len(self.train_dataloader.dataset)
        batches = math.ceil(dataset_len / self.train_dataloader.batch_size)

        with tqdm(total=batches) as prog:
            for X_train, y_train in self.train_dataloader:
                X_train, y_train = X_train.to(self.device), y_train.to(self.device)
                logits = self.model(X_train)
                batch_loss = self.loss_fn(logits, y_train)

                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

                class_pred = torch.argmax(logits, dim=-1).detach()
                class_train = torch.argmax(y_train, dim=-1).detach()

                correct += torch.sum(class_pred == class_train)
                loss += batch_loss.detach()
                prog.update(1)

        if self.lr_scheduler:
            self.lr_scheduler.step()

        loss = loss.item() / batches
        acc = correct.item() / dataset_len

        self.train_acc.append(acc)
        self.train_loss.append(loss)

    def _validate(self):
        loss = torch.tensor(0.0).to(self.device)
        correct = torch.tensor(0.0).to(self.device)

        dataset_len = len(self.val_dataloader.dataset)
        batches = math.ceil(dataset_len / self.val_dataloader.batch_size)
        for X_val, y_val in self.val_dataloader:
            X_val, y_val = X_val.to(self.device), y_val.to(self.device)
            logits = self.model.infer(X_val)
            batch_loss = self.loss_fn(logits, y_val)

            class_pred = torch.argmax(logits, dim=-1)
            class_val = torch.argmax(y_val, dim=-1)

            correct += torch.sum(class_pred == class_val)
            loss += batch_loss

        loss = loss.item() / batches
        acc = correct.item() / dataset_len

        self.val_acc.append(acc)
        self.val_loss.append(loss)

    def train(self):
        trained_epoches = 0 if not self.train_epoches else self.train_epoches[-1]
        for epoch in range(trained_epoches + 1, trained_epoches + 1 + self.epoches):
            print("-" * 20 + " " + f"epoch {epoch}" + " " + "-" * 20)
            self._train_one_epoch()
            self.train_epoches.append(epoch)
            print(f"train loss: {self.train_loss[-1]: .4f}")
            print(f"train accuracy: {self.train_acc[-1]: .2%}")

            if (epoch == 1 or epoch % self.test_every == 0) and self.val_dataloader and self.test_every:
                print("testing the model...")
                self._validate()
                self.val_epoches.append(epoch)
                print(f"validation loss: {self.val_loss[-1]: .4f}")
                print(f"validation accuracy: {self.val_acc[-1]: .2%}")
            print("-" * 51)

            if self.save_every:
                path = None
                if epoch % self.save_every == 0 and self.save_every != -1:
                    path = Path(self.root / f"{self.model.name}_epoch{epoch}.pt").resolve()
                    self.root.mkdir(parents=True, exist_ok=True)
                    torch.save(self.model, path)
                if epoch == trained_epoches + self.epoches:
                    path = Path(self.root / f"{self.model.name}_final.pt").resolve()
                    self.root.mkdir(parents=True, exist_ok=True)
                    torch.save(self.model, path)
