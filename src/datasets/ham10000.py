from typing import Callable, Optional, Tuple
import pickle

from torch.utils.data import Dataset, Subset
import numpy as np
import pandas as pd
from torchvision.datasets.folder import default_loader
from sklearn.model_selection import train_test_split
from torchvision import transforms

from utils.file_path_generator import load_from_file, save_to_file, stats_file
from datasets.utils import calc_mean_std


IDENTITY = lambda _: _


# adjusted from https://github.com/a1302z/ObjaxDPTraining/blob/derma/dptraining/datasets/ham10000.py
class HAM10000(Dataset):

    def __init__(
        self,
        data_dir,
        metadata: Optional[pd.DataFrame] = None,
        merge_labels: bool = True,
        transform: Optional[Callable] = None,
        label_transform: Optional[Callable] = None,
        seed=1,
    ) -> None:
        super().__init__()
        self.data = data_dir
        self.metadata = (
            pd.read_csv(data_dir / "HAM10000_metadata.csv")
            if metadata is None
            else metadata
        )
        self.imgs = {
            img.stem: img
            for img in (data_dir / "HAM10000_images").rglob("*.jpg")
            if img.is_file()
        }
        if merge_labels:
            # seperated in needs attention vs no urgent attention required
            self.metadata["label"] = (
                self.metadata["dx"].isin(["akiec", "bcc", "mel"]).astype(int)
            )
        else:
            label_assignment = {val: i for i, val in enumerate(df.dx.unique())}
            self.metadata["label"] = self.metadata.dx.map(label_assignment).astype(int)

        self.transform = transform if transform is not None else IDENTITY
        self.label_transform = (
            label_transform if label_transform is not None else IDENTITY
        )

    def __getitem__(self, index: int):
        entry = self.metadata.iloc[index]
        img_name = self.imgs[entry.image_id]
        label = entry.label
        img = default_loader(img_name)

        img = self.transform(img)
        label = self.label_transform(label)

        return img, label

    def __len__(self) -> int:
        return len(self.metadata)

    def get_labels(self) -> list:
        return list(self.metadata['label'])

    def stratified_split(self, train_size, test_size):
        labels = self.get_labels()
        indices = list(range(len(labels)))

        indices_train_val, indices_test, labels_train_val, _ = train_test_split(
            indices, labels, stratify=labels, test_size=test_size, random_state=0
            )

        indices_train, indices_val, _, _ = train_test_split(
            indices_train_val, labels_train_val, stratify=labels_train_val, train_size=train_size, random_state=0
            )

        train = Subset(self, indices_train)
        val = Subset(self, indices_val)
        test = Subset(self, indices_test)

        return train, val, test


class DataLoader():
    def __init__(self, data_dir):
        self.data_dir = data_dir

        self._train_data = None
        self._val_data = None
        self._test_data = None

    @property
    def train_data(self, config):
        if self._train_data == None:
            self.make_datasets(config)
        return self._train_data

    @property
    def val_data(self, config):
        if self._val_data == None:
            self.make_datasets(config)
        return self._val_data

    @property
    def test_data(self, config):
        if self._test_data == None:
            self.make_datasets(config)
        return self._test_data

    def get_mean_std(self, config):
        if stats_file(config).is_file():
            mean, std = load_from_file(stats_file(config))

        else:
            tf = transforms.Compose([
            transforms.Resize(224), 
            transforms.CenterCrop(224), 
            transforms.ToTensor(),
            ])
            
            ds = HAM10000(data_dir=self.data_dir, transform=tf)
            train, _, _ = ds.stratified_split(train_size=config.dataset.train_size, test_size=config.dataset.test_size)
            mean, std = calc_mean_std(train)

            save_to_file(info=(mean, std), file_path=stats_file(config))

        return mean, std

    def make_datasets(self, config):
        """ 
        Returns standardized training, validation and test data.
        Stratified split is used to get train, val and test datasets.
        """
        mean, std = self.get_mean_std(config)
        
        basic_transform = transforms.Compose([
            transforms.Resize(224), 
            transforms.CenterCrop(224), 
            transforms.ToTensor(),
            transforms.Normalize(mean, std) 
        ])

        ds = HAM10000(data_dir=self.data_dir, transform=basic_transform)

        train, val, test = ds.stratified_split(train_size=config.dataset.train_size, test_size=config.dataset.test_size)

        self._train_data = train
        self._val_data = val
        self._test_data = test

        return train, val, test