from typing import Tuple
import random
import pickle

from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from torch import manual_seed
from torch.utils.data import Dataset, Subset
from torchvision import transforms
import numpy as np
import pandas as pd
import torch

from utils.file_path_generator import load_from_file, save_to_file, stats_file
from datasets.utils import calc_mean_std

# TODO if binary classification: binary before classification/ im zug daten fuer klassifizierung "vorbereiten" -> allows to extract features only once
# adjusted from https://github.com/gkaissis/PriMIA/blob/master/torchlib/dataloader.py
class PPPP(Dataset):
    
    def __init__(
        self, 
        data_dir,
        train = False,
        single_channel = True,
        transform = None, 
        seed = 1,
    ):
        super().__init__()
        random.seed(seed)
        manual_seed(seed)
        self.train = train
        self.data_dir = data_dir
        self.label_df = pd.read_csv(data_dir/'Labels.csv')
        self.labels = self.label_df[
            self.label_df["Dataset_type"] == ("TRAIN" if train else "TEST")
        ]
        self.single_channel = single_channel
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        row = self.labels.iloc[index]
        label = torch.tensor(row["Numeric_Label"])
        
        infection_type = 'normal'
        if 'bacteria' in row['X_ray_image_name']:
            infection_type = 'bacterial pneumonia'
        elif 'virus' in row['X_ray_image_name']:
            infection_type = 'viral pneumonia'
        path =  self.data_dir / ('train' if self.train else 'test') / infection_type / row["X_ray_image_name"]
        img = Image.open(path)

        if self.single_channel:
            img = ImageOps.grayscale(img)

        if self.transform:
            img = self.transform(img)

        return img, label

    def get_labels(self) -> np.ndarray:
        return np.array(self.labels["Numeric_Label"])

    def stratified_split(self, train_size=0.9):
        labels = self.get_labels()
        indices = list(range(len(labels)))

        indices_train, indices_val, labels_train, labels_val = train_test_split(
                indices, labels, stratify=labels, train_size=train_size, random_state=0
                )
        
        train = Subset(self, indices_train)
        val = Subset(self, indices_val)

        return train, val


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

            train_val = PPPP(train=True, data_dir=self.data_dir, transform=tf)
            train, _ = train_val.stratified_split(train_size=config.dataset.train_size)
            mean, std = calc_mean_std(train)

            save_to_file(data=(mean, std), file_path=stats_file(config))

        return mean, std

    def make_datasets(self, config):
        """ 
        Returns standardized training, validation and test data.
        Stratified split is used to get train and val datasets.
        """
        mean, std = self.get_mean_std(config)
        
        basic_transform = transforms.Compose([
            transforms.Resize(224), 
            transforms.CenterCrop(224), 
            transforms.ToTensor(),
            transforms.Normalize(mean, std) 
        ])

        train_val = PPPP(train=True, data_dir=self.data_dir, transform=basic_transform)
        test = PPPP(train=False, data_dir=self.data_dir, transform=basic_transform)

        train, val = train_val.stratified_split(train_size=config.dataset.train_size)

        self._train_data = train
        self._val_data = val
        self._test_data = test

        return train, val, test