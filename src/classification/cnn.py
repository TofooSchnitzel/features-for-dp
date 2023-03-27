from abc import ABC, abstractmethod
from time import sleep

from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import torch
import wandb

from classification.models import ResNet9
from classification.utils import get_data_from_dict


class TorchDataModule(Dataset):
    def __init__(self, feature_dict: dict, make_binary: bool = False, flatten: bool = False):
        self.feature_dict = feature_dict
        self.make_binary = make_binary
        self.flatten = False
    
    def make_datasets(self):
        X_train, y_train, X_val, y_val, X_test, y_test = get_data_from_dict(self.feature_dict, flatten=self.flatten)

        if self.make_binary:
            y_train, y_val, y_test = [torch.where(data==2, 1, data) for data in [y_train, y_val, y_test]]

        train = torch.utils.data.TensorDataset(X_train, y_train)
        val = torch.utils.data.TensorDataset(X_val, y_val)
        test = torch.utils.data.TensorDataset(X_test, y_test)

        img = train[0][0]
        self.num_channels = img.shape[0]
        self.num_classes = len(np.unique(y_train))

        return train, val, test


class CNNTrainer(ABC):
    def __init__(self, 
                feature_dict, 
                make_binary: bool = False, 
                batch_size: int = 32, 
                epochs: int = 10, 
                lr: float = 0.001,
                scale_norm: bool = False, 
                log_wandb: bool = False,
                seed: int = 0,
                ):
        data = TorchDataModule(feature_dict, make_binary)
        self.train, self.val, self.test = data.make_datasets()
        self.num_classes = data.num_classes
        self.num_channels = data.num_channels

        self.batch_size = batch_size
        self.epochs = epochs
        self.log_wandb = log_wandb

        self.model = ResNet9(in_channels=self.num_channels, num_classes=self.num_classes, scale_norm=scale_norm)
        self.optimizer = self.configure_optimizer(model=self.model ,lr=lr)
        self.criterion = self.configure_criterion()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)

    def configure_optimizer(self, model, lr):
        return torch.optim.NAdam(model.parameters(), lr=lr)
    
    def configure_criterion(self):
        if self.num_classes == 2:
            return torch.nn.BCELoss()
        else: 
            return torch.nn.CrossEntropyLoss()
    
    def configure_dataloader(self, data: torch.utils.data.TensorDataset, shuffle: bool = False):
        return torch.utils.data.DataLoader(data, batch_size=self.batch_size, shuffle=shuffle)
        
    def fit(self):
        trainloader = self.configure_dataloader(self.train, shuffle=True)

        self.model.to(self.device)
        self.model.train()

        losses = []

        for epoch in range(self.epochs):  
            with tqdm(trainloader, unit="batch") as tepoch:
                for img, labels in tepoch:
                    tepoch.set_description(f"Epoch {epoch}/{self.epochs}")
                    
                    self.optimizer.zero_grad()
                    outputs = self.model(img.to(self.device))
                    
                    if self.num_classes == 2:
                        labels = labels.unsqueeze(1).float()

                    loss = self.criterion(outputs, labels.to(self.device))
                    losses.append(loss.item())
                    loss.backward()
                    self.optimizer.step()

                    if self.log_wandb:
                        wandb.log({"loss": np.mean(losses)})
                    
                    tepoch.set_postfix(loss=loss.item())
                    sleep(0.1)

    def predict(self, predict_on_test: bool = False):
        y_true = []
        y_pred = []

        data = self.val if not predict_on_test else self.test
        loader = self.configure_dataloader(data, shuffle=False)

        self.model.eval()
        for data in loader:
            images, labels = data
            outputs = self.model(images.to(self.device))

            if self.num_classes == 2:
                preds = ((outputs.detach().cpu().numpy()) >= 0.5)
            else:
                preds = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            labels = labels.detach().cpu().numpy()

            y_true = np.concatenate((y_true, labels), axis=None)
            y_pred = np.concatenate((y_pred, preds), axis=None)
        
        return y_true, y_pred
    

class DPCNNTrainer(CNNTrainer):
    def __init__(self, feature_dict, make_binary: bool = False, 
                batch_size: int = 32, epochs: int = 10, lr: float = 0.001,
                scale_norm: bool = False, log_wandb: bool = False, seed: int = 0,
                max_grad_norm: float = 1.2, epsilon: float = 1.0, delta: float = 1e-5, max_physical_batch_size: int = 32,
                ):
        super().__init__(feature_dict=feature_dict, make_binary=make_binary, batch_size=batch_size, epochs=epochs, lr=lr,
                         scale_norm=scale_norm, log_wandb=log_wandb, seed=seed)
        self.max_grad_norm = max_grad_norm
        self.epsilon = epsilon
        self.delta = delta
        self.max_physical_batch_size = max_physical_batch_size
        self.privacy_engine = PrivacyEngine()

    def configure_optimizer(self, model, lr):
        return super().configure_optimizer(model=model, lr=lr)
    
    def configure_criterion(self):
        return super().configure_criterion()
    
    def configure_dataloader(self, data: torch.utils.data.TensorDataset, shuffle: bool = False):
        return super().configure_dataloader(data, shuffle=shuffle)
    
    def prepare_private_training(self):
        private_model, private_optimizer, private_trainloader = self.privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.configure_dataloader(data=self.train, shuffle=True),
            epochs=self.epochs,
            max_grad_norm=self.max_grad_norm,
            target_epsilon=self.epsilon,
            target_delta=self.delta,
        )
        print(f"Using sigma={private_optimizer.noise_multiplier} and C={self.max_grad_norm}")

        return private_model, private_optimizer, private_trainloader
        
    def fit(self):
        private_model, private_optimizer, private_trainloader = self.prepare_private_training()

        private_model.to(self.device)
        private_model.train()
        for epoch in range(self.epochs):

            losses = []

            with BatchMemoryManager(
                    data_loader = private_trainloader,
                    max_physical_batch_size = self.max_physical_batch_size,
                    optimizer = private_optimizer,
                ) as memory_safe_data_loader:
                    with tqdm(memory_safe_data_loader, unit="batch") as tepoch:
                    
                        for imgs, labels in tepoch:
                            tepoch.set_description(f"Epoch {epoch}/{self.epochs}")
                            
                            private_optimizer.zero_grad()
                            outputs = private_model(imgs.to(self.device))

                            if self.num_classes == 2:
                                labels = labels.unsqueeze(1).float()

                            loss = self.criterion(outputs, labels.to(self.device))
                            losses.append(loss.item())
                            loss.backward()

                            if self.log_wandb:
                                wandb.log({"loss": np.mean(losses)})

                            private_optimizer.step()

                            tepoch.set_postfix(loss=loss.item())
                            sleep(0.1)

        self.private_model = private_model

    def predict(self, predict_on_test: bool = False):
        y_true = []
        y_pred = []

        data = self.val if not predict_on_test else self.test
        loader = self.configure_dataloader(data, shuffle=False)

        self.private_model.eval()
        for data in loader:
            images, labels = data
            outputs = self.private_model(images.to(self.device))

            preds = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            labels = labels.detach().cpu().numpy()

            y_true = np.concatenate((y_true, labels), axis=None)
            y_pred = np.concatenate((y_pred, preds), axis=None)
        
        return y_true, y_pred