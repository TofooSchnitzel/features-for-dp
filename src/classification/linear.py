from time import sleep

from tqdm import tqdm
import torch
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator
import numpy as np
import wandb

from classification.cnn import TorchDataModule
from classification.models import Linear
from classification.utils import get_data_from_dict


class DPLinearTrainer():
    def __init__(self, feature_dict, make_binary: bool = False,
                 batch_size: int = 32, epochs: int = 10, lr: float = 0.001,
                 epsilon: float = 1.0, delta: float = 1e-5, max_grad_norm: float = 1.0, max_physical_batch_size: int = 128,
                 optimizer_name: str = "nadam", log_wandb: bool = False, seed: int = 0):
        self.make_binary = make_binary
        self.log_wandb = log_wandb

        X_train, y_train, X_val, y_val, X_test, y_test = get_data_from_dict(feature_dict, flatten=True)
        img = self.train[0][0]
        self.num_channels = img.shape[0]
        self.num_classes = len(np.unique(y_train))
        
        # flat = True if len(feature_dict["train"]["features"][0]) == 1 else False
        # if not flat:      # data is not flattened yet

            
        #     X_train, y_train, X_val, y_val, X_test, y_test = get_data_from_dict(feature_dict)
        #     X_train = (torch.flatten(self.featy, start_dim=1))
        #     X_val = (torch.flatten(X_val, start_dim=1))
        #     X_test = (torch.flatten(X_test, start_dim=1))

        #     if self.make_binary:
        #         y_train, y_val, y_test = [torch.where(data==2, 1, data) for data in [y_train, y_val, y_test]]
            
        #     self.train = torch.utils.data.TensorDataset(X_train, y_train)
        #     self.val = torch.utils.data.TensorDataset(X_val, y_val)
        #     self.test = torch.utils.data.TensorDataset(X_test, y_test)

        #     img = self.train[0][0]
        #     self.num_channels = img.shape[0]
        #     self.num_classes = len(np.unique(y_train))
        # else:
        #     data = TorchDataModule(feature_dict, self.make_binary)
        #     self.train, self.val, self.test = data.make_datasets()

        #     self.num_classes = data.num_classes
        #     self.num_channels = data.num_channels

        self.batch_size = batch_size
        self.epochs = epochs

        self.model = Linear(in_channels=self.num_channels, num_classes=self.num_classes)
        self.optimizer = self.configure_optimizer(optimizer_name=optimizer_name, model=self.model ,lr=lr)
        self.criterion = self.configure_criterion()

        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.max_physical_batch_size = max_physical_batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)

    def configure_optimizer(self, optimizer_name, model, lr):
        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == "nadam":
            optimizer = torch.optim.NAdam(model.parameters(), lr=lr)
        elif optimizer_name == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        else:
            raise ValueError(f"Invalid optimizer name: {optimizer_name}")
        return optimizer
    
    def configure_criterion(self):
        if self.num_classes == 2:
            return torch.nn.BCELoss()
        else: 
            return torch.nn.CrossEntropyLoss()
    
    def configure_dataloader(self, data: torch.utils.data.TensorDataset, shuffle: bool = False):
        return torch.utils.data.DataLoader(data, batch_size=self.batch_size, shuffle=shuffle)

    def prepare_private_training(self):
        private_model, private_optimizer, private_trainloader = PrivacyEngine().make_private_with_epsilon(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.configure_dataloader(data=self.train, shuffle=True),
            epochs=self.epochs,
            max_grad_norm=self.max_grad_norm,
            target_epsilon=self.epsilon,
            target_delta=self.delta,
        )

        errors = ModuleValidator.validate(private_model, strict=False)
        if errors:
            print(errors)
            private_model = ModuleValidator.fix(private_model)
            errors = ModuleValidator.validate(private_model, strict=False)
            if errors: print(errors)

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
                    
                        for img, labels in tepoch:
                            tepoch.set_description(f"Epoch {epoch}/{self.epochs}")
                            
                            private_optimizer.zero_grad()
                            outputs = private_model(img.to(self.device))
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