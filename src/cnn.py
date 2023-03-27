from omegaconf import DictConfig
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
import hydra
import numpy as np
import random
import torch
import wandb

import torch.nn.modules.module

from utils.config import create_config_wandb
from datasets import select_dataloader
from classification.models import ResNet9

import warnings
warnings.simplefilter("ignore")


def predict(private_model, dataloader, device):

    y_true = []
    y_pred = []

    private_model.eval()
    for data in dataloader:
        images, labels = data
        outputs = private_model(images.to(device))

        preds = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        labels = labels.detach().cpu().numpy()

        y_true = np.concatenate((y_true, labels), axis=None)
        y_pred = np.concatenate((y_pred, preds), axis=None)

    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    conf = confusion_matrix(y_true, y_pred)

    return acc, mcc, conf

@hydra.main(version_base=None, config_path='../config/reproduce_results', config_name='cnn.yaml')
def end_to_end(config: DictConfig):
    for seed in config.seeds:     # multiple runs
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        if config.log_wandb:
            project_name, config_wandb = create_config_wandb(config)

            run = wandb.init(
                project=project_name,
                config=config_wandb,
                settings=wandb.Settings(start_method="thread"),
                reinit=True,
            )

        dataloader = select_dataloader(config)
        train, val, test = dataloader.make_datasets(config)

        trainloader = DataLoader(train, batch_size=config.classifier.batch_size, shuffle=True)
        valloader = DataLoader(val, batch_size=config.classifier.batch_size, shuffle=False)
        testloader = DataLoader(test, batch_size=config.classifier.batch_size, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)

        num_channels = train[0][0].shape[0]
        num_classes = len(config.dataset.classes)

        model = ResNet9(
            in_channels=num_channels,
            num_classes=num_classes,
            scale_norm=config.classifier.scale_norm,
            )
        
        privacy_engine = PrivacyEngine()
        
        errors = ModuleValidator.validate(model, strict=False)
        if errors:
            print(errors)
            model = ModuleValidator.fix(model)
            errors = ModuleValidator.validate(model, strict=False)
            if errors: print(errors)

        model.to(device)
        
        if num_classes == 2:
            criterion = torch.nn.BCELoss()
        else: 
            criterion = torch.nn.CrossEntropyLoss()

        if config.classifier.optimizer_name == "nadam":
            optimizer = torch.optim.NAdam(model.parameters(), lr=config.classifier.lr)
        else: 
            raise NotImplementedError

        model, optimizer, trainloader = privacy_engine.make_private_with_epsilon(
            module = model,
            optimizer = optimizer,
            data_loader = trainloader,
            epochs = config.classifier.epochs,
            max_grad_norm = config.classifier.max_grad_norm,
            target_epsilon = config.classifier.epsilon,
            target_delta = config.classifier.delta,
        )
        
        for epoch in tqdm(range(config.classifier.epochs), desc="Epoch", unit="epoch"):
            model.train()

            losses = []
            mcc_train = []

            with BatchMemoryManager(
                data_loader=trainloader, 
                max_physical_batch_size=config.classifier.max_physical_batch_size, 
                optimizer=optimizer
                ) as memory_safe_data_loader:

                with tqdm(memory_safe_data_loader, unit="batch") as tepoch:

                    for images, targets in tepoch: 
                        tepoch.set_description(f"Epoch {epoch+1}/{config.classifier.epochs}")
                          
                        optimizer.zero_grad()
                        images = images.to(device)
                        targets = targets.to(device)

                        # compute output
                        output = model(images)
                        loss = criterion(output, targets)

                        preds = np.argmax(output.detach().cpu().numpy(), axis=1)
                        labels = targets.detach().cpu().numpy()

                        # measure accuracy and record loss
                        mcc = matthews_corrcoef(preds, labels)

                        losses.append(loss.item())
                        mcc_train.append(mcc)
                        
                        if config.log_wandb:
                            wandb.log({"loss": np.mean(losses)})

                        loss.backward()
                        optimizer.step()
                    
                    if config.log_wandb:
                        wandb.log({"train_acc": np.mean(mcc_train)})

        acc, mcc, conf = predict(model, valloader, device)

        if config.log_wandb:
            wandb.log({
                "seed": seed,
                "val_acc": acc,
                "val_mcc": mcc,
                "val_confusion_matrix": conf,
                })
            
        else: 
            print(f"Results on val set")
            print(f"Accuracy: {round(acc, 4)}")
            print(f"Matthews Correlation Coef: {round(mcc, 4)}")
            print(f"Confusion Matrix: \n{conf}")

        if config.evaluation:
            acc, mcc, conf = predict(model, testloader, device)

            if config.log_wandb:
                wandb.log({
                    "seed": seed,
                    "test_acc": acc,
                    "test_mcc": mcc,
                    "test_confusion_matrix": conf,
                    })
            
            else:
                print(f"Results on test set")
                print(f"Accuracy: {round(acc, 4)}")
                print(f"Matthews Correlation Coef: {round(mcc, 4)}")
                print(f"Confusion Matrix: \n{conf}")



if __name__ == "__main__":
    end_to_end()