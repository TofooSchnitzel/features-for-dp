from omegaconf import OmegaConf
import hydra
import numpy as np
import random
import torch
import wandb
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix

from classification import prepare_training
from datasets import select_dataloader
from feature_extraction import select_feature_extractor
from utils.config import check_config_is_valid, project_name

@hydra.main(version_base=None, config_path='../config/reproduce_results', config_name='sn_gbdt.yaml')
def fixed_features(config):
    for seed in config.seeds:     # multiple runs
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        check_config_is_valid(config)

        if config.log_wandb:
            run = wandb.init(
                project=project_name(config),
                config=OmegaConf.to_container(config, resolve=True),
                settings=wandb.Settings(start_method="thread"),
                reinit=True,
            )

        dataloader = select_dataloader(config)
        train, val, test = dataloader.make_datasets(config)

        feature_extractor = select_feature_extractor(config)
        feature_dict = feature_extractor.get_features_as_dict(train, val, test, config)

        trainer = prepare_training(config, feature_dict, seed=seed)
        trainer.fit()
        y_true, y_pred = trainer.predict()

        acc = accuracy_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        conf = confusion_matrix(y_true, y_pred)

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
            y_true, y_pred = trainer.predict(predict_on_test=True)

            acc = accuracy_score(y_true, y_pred)
            mcc = matthews_corrcoef(y_true, y_pred)
            conf = confusion_matrix(y_true, y_pred)

            if config.log_wandb:
                wandb.log({
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
    fixed_features()