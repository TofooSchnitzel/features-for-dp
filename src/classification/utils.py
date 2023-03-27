from typing import Tuple

import torch
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MinMaxScaler


def get_data_from_dict(feature_dict: dict, flatten: bool) -> Tuple[torch.Tensor, ...]:
    img = feature_dict["train"]["features"][0]
    is_flat = True if len(img.shape) == 1 else False

    if not is_flat and flatten:
        X_train = torch.flatten(torch.stack(feature_dict['train']['features'], dim=0), start_dim=1)
        X_val = torch.flatten(torch.stack(feature_dict['val']['features'], dim=0), start_dim=1)
        X_test = torch.flatten(torch.stack(feature_dict['test']['features'], dim=0), start_dim=1)

    else:
        k, c, w, h = img.shape
        X_train = (torch.stack(feature_dict['train']['features'], dim=0)).view(-1, k*c, w, h)
        X_val = (torch.stack(feature_dict['val']['features'], dim=0)).view(-1, k*c, w, h)
        X_test = (torch.stack(feature_dict['test']['features'], dim=0)).view(-1, k*c, w, h)

    y_train = torch.tensor(feature_dict['train']['labels'])
    y_val = torch.tensor(feature_dict['val']['labels'])
    y_test = torch.tensor(feature_dict['test']['labels'])

    return X_train, y_train, X_val, y_val, X_test, y_test


def calculate_sample_weights(y_train: np.ndarray) -> np.ndarray:
    assert len(np.unique(y_train)) == 2     # only implemented for binary labels

    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
    sample_weights = np.where(y_train==0, class_weights[0], class_weights[1])

    return sample_weights


def normalize_and_clip_data(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray, feature_range = (0,1),
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fits the scaler on the training data.
    Returns the scaled and clipped train, val, and test data.
    """
    scaler = MinMaxScaler(feature_range=feature_range)

    X_train = scaler.fit_transform(X_train)
    X_val, X_test = [scaler.transform(data) for data in [X_val, X_test]]
    
    X_train, X_val, X_test = [np.clip(data, a_min=feature_range[0], a_max=feature_range[1]) for data in [X_train, X_val, X_test]]

    return X_train, X_val, X_test   


def undersample_data(X_train: np.ndarray, y_train: np.ndarray, seed: int = 0,
    ) -> Tuple[np.ndarray,np.ndarray]:
    """
    Reduces the number of samples of every class 
    so they match the number of samples of the smalles class.
    """
    unique, count = np.unique(y_train, return_counts=True)

    idcs_dict = dict()

    for label in unique:
        idcs = np.where(y_train==label)[0]
        idcs_dict[label] = idcs
    
    num_samples_minority_class = min(count)

    rng = np.random.default_rng(seed=seed)
    selected_idcs = []

    for label in idcs_dict.keys():
        idcs = np.where(y_train==label)[0]
        rng.shuffle(idcs)
        idcs = idcs[0:num_samples_minority_class]
        selected_idcs = np.append(selected_idcs, idcs).astype(int)

    X_train = X_train[selected_idcs]
    y_train = y_train[selected_idcs]

    return X_train, y_train