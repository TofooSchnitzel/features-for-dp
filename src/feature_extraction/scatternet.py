from typing import Tuple

import numpy as np
import torch
from kymatio.torch import Scattering2D
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.file_path_generator import load_from_file, save_to_file, feature_file


class FeatureExtractor():
    def __init__(self, J: int = 6, L: int = 8, m: int = 1):
        self.J = J
        self.L = L
        self.m = m

    def extract_features(self, dataset: Dataset, shape: Tuple[int, int] = (224,224)) -> dict:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        scattering = Scattering2D(J=self.J, L=self.L, max_order=self.m, shape=shape).to(device)

        features = []
        labels = []

        for (img, label) in tqdm(dataset, total=len(dataset), leave=False, desc="extract features"):
            img = img.to(device)
            feature = scattering(img)
            features.append(feature)
            labels.append(label)

        return dict(features=features, labels=labels)

    def get_features_as_dict(self, train: Dataset, val: Dataset, test: Dataset, config) -> dict:
        if feature_file(config).is_file():
            return load_from_file(feature_file(config))
        
        else: 
            train = self.extract_features(train)
            val = self.extract_features(val)
            test = self.extract_features(test)

            feature_dict = dict(train=train, val=val, test=test)

        if config.save_features:
            save_to_file(data=feature_dict, path=feature_file(config))

        return feature_dict