from enum import Enum
from pathlib import Path

from objax import nn, io
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import torch

from feature_extraction.model import ResNet9
from utils.file_path_generator import load_from_file, save_to_file, feature_file

WEIGHTS_PATH = Path(__file__).parent/'weights'


class ModelType(Enum):
    RadimageNet = 'radimagenet',
    ImageNet = 'imagenet'


class FeatureExtractor():
    def __init__(self, modeltype: str):
        self.modeltype = modeltype

    def load_pretrained_model(self):
        if self.modeltype == 'imagenet':
            in_channels = 3
            num_classes = 1000
            weights_file = WEIGHTS_PATH/'imagenet_resnet9_gn_mp.npz'

        elif self.modeltype == 'radimagenet':
            in_channels = 1
            num_classes = 165
            weights_file = WEIGHTS_PATH/'radimagenet_resnet9_gn_maxpool.npz'

        else:
            raise ValueError(f'Modeltype ({self.modeltype}) not supported')

        model = ResNet9(in_channels=in_channels, num_classes=num_classes, norm_cls=nn.GroupNorm2D)
        io.load_var_collection(weights_file, model.vars())

        return model

    def extract_features(self, dataset: Dataset) -> dict:
        model = self.load_pretrained_model()

        features = []
        labels = []

        for data in tqdm(dataset, total=len(dataset), leave=False, desc='extracting features'):
            img = np.array(data[0])
            label = data[1]

            c, w, h = img.shape
            img = np.reshape(img, (1,c,w,h))    # reshape with np is faster

            feature = model.feature_extractor(img, training=False)

            features.append(np.array(feature[0]))
            labels.append(label)

        features = torch.tensor(np.array(features))

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