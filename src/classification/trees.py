from abc import ABC, abstractmethod
import warnings

import torch
import numpy as np
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix
from tabpfn import TabPFNClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from interpret.privacy import DPExplainableBoostingClassifier

from classification.utils import get_data_from_dict, calculate_sample_weights, normalize_and_clip_data, undersample_data
from utils.file_path_generator import load_from_file, save_to_file
from federated_gbdt.models.gbdt.private_gbdt import PrivateGBDT
# from federated_gbdt_old.models.gbdt.private_gbdt import PrivateGBDT


class TreeTrainer(ABC):
    def __init__(self, feature_dict: dict, make_binary: bool = False, pca: float = 1.0,   # pca = 1.0 means no pca
            pca_file: str = "", seed: int = 0):
        self.make_binary = make_binary
        self.pca = pca
        self.pca_file = pca_file
        self.seed = seed

        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = self.prepare_data(feature_dict)

    @abstractmethod
    def prepare_data(self, feature_dict: dict):
        if self.pca < 1.0:
            if self.pca_file.is_file():
                print("pca")
                feature_dict = load_from_file(self.pca_file)
                X_train, y_train, X_val, y_val, X_test, y_test = get_data_from_dict(feature_dict, flatten=True)
            else:
                X_train, y_train, X_val, y_val, X_test, y_test = get_data_from_dict(feature_dict, flatten=True)
                X_train = X_train.detach().cpu()
                X_val = X_val.detach().cpu()
                X_test = X_test.detach().cpu()

                pca = PCA(n_components=self.pca, random_state=self.seed)
                X_train = pca.fit_transform(X_train)
                X_val = pca.transform(X_val)
                X_test = pca.transform(X_test)

                feature_dict["train"]["features"] = X_train
                feature_dict["val"]["features"] = X_val
                feature_dict["test"]["features"] = X_test

                save_to_file(data=feature_dict, path=self.pca_file)    # save pca to file

        else:
            X_train, y_train, X_val, y_val, X_test, y_test = get_data_from_dict(feature_dict, flatten=True)
            X_train = X_train.detach().cpu()
            X_val = X_val.detach().cpu()  
            X_test = X_test.detach().cpu()

        if self.make_binary:
            y_train, y_val, y_test = [torch.where(data==2, 1, data) for data in [y_train, y_val, y_test]]

        return X_train, y_train, X_val, y_val, X_test, y_test


class LinearTrainer(TreeTrainer):
    def __init__(self, feature_dict: dict, make_binary: bool = False,
                 pca: float = 1.0, pca_file: str = "", seed: int = 0):
        super().__init__(feature_dict, make_binary, pca, pca_file, seed)

    def prepare_data(self, feature_dict: dict):
        X_train, y_train, X_val, y_val, X_test, y_test = super().prepare_data(feature_dict)

        return X_train, y_train, X_val, y_val, X_test, y_test

    def fit(self):
        classifier = LogisticRegression(random_state=self.seed)
        self.fitted_classifier = classifier.fit(self.X_train, self.y_train)

    def predict(self, predict_on_test: bool = False):
        if predict_on_test:
            y_pred = self.fitted_classifier.predict(self.X_test)
            y_true = self.y_test
        else:
            y_pred = self.fitted_classifier.predict(self.X_val)
            y_true = self.y_val
        return y_true, y_pred


class GBDTTrainer(TreeTrainer):
    def __init__(self, feature_dict: dict, make_binary: bool = False, n_estimators: int = 100,
                 pca: float = 1.0, pca_file: str = "", seed: int = 0):
        super().__init__(feature_dict, make_binary, pca, pca_file, seed)
        self.n_estimators = n_estimators

    def prepare_data(self, feature_dict: dict):
        X_train, y_train, X_val, y_val, X_test, y_test = super().prepare_data(feature_dict)

        return X_train, y_train, X_val, y_val, X_test, y_test

    def fit(self):
        # PrivateGBDT should work with epsilon=0 as non-private variant, however "Segmentation fault (core dumped)" error occurs
        # thus we utilize sklearn's GradientBoostingClassifier as approximation
        classifier = GradientBoostingClassifier(n_estimators=self.n_estimators, random_state=self.seed)     
        self.fitted_classifier = classifier.fit(self.X_train, self.y_train)
    
    def predict(self, predict_on_test: bool = False):
        if predict_on_test:
            y_pred = self.fitted_classifier.predict(self.X_test)
            y_true = self.y_test
        else:
            y_pred = self.fitted_classifier.predict(self.X_val)
            y_true = self.y_val
        return y_true, y_pred


class EBMTrainer(TreeTrainer):
    def __init__(self, feature_dict: dict, make_binary: bool = False,  max_rounds: int = 5000,
                 pca: float = 1.0, pca_file: str = "", seed: int = 0):
        super().__init__(feature_dict, make_binary, pca, pca_file, seed)
        self.max_rounds = max_rounds

    def prepare_data(self, feature_dict: dict):
        X_train, y_train, X_val, y_val, X_test, y_test = super().prepare_data(feature_dict)

        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def fit(self):
        classifier = ExplainableBoostingClassifier(random_state=self.seed, max_rounds=self.max_rounds)
        self.fitted_classifier = classifier.fit(self.X_train, self.y_train)

    def predict(self, predict_on_test: bool = False):
        if predict_on_test:
            y_pred = self.fitted_classifier.predict(self.X_test)
            y_true = self.y_test

        else:
            y_pred = self.fitted_classifier.predict(self.X_val)
            y_true = self.y_val

        return y_true, y_pred


class TabPFNTrainer(TreeTrainer):
    def __init__(self, feature_dict: dict, make_binary: bool = False, 
                 pca: float = 1.0, pca_file: str = "", seed: int = 0):
        super().__init__(feature_dict, make_binary, pca, pca_file, seed)

    def prepare_data(self, feature_dict: dict):
        return super().prepare_data(feature_dict)
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray):
        return super().evaluate(y_true, y_pred)

    def fit(self):
        self.fitted_classifier = TabPFNClassifier(seed=self.seed).fit(self.X_train, self.y_train, overwrite_warning=True)


    def predict(self, predict_on_test: bool = False):
        if predict_on_test:
            y_pred = self.fitted_classifier.predict(self.X_test)
            y_true = self.y_test
        else:
            y_pred = self.fitted_classifier.predict(self.X_val)
            y_true = self.y_val

        return y_true, y_pred
    

class DPEBMTrainer(TreeTrainer):
    def __init__(self, 
                 feature_dict: dict, 
                 make_binary: bool = False, 
                 pca: float = 1.0, 
                 pca_file: str = "", 
                 seed: int = 0,
                 epsilon: float = 1.0, 
                 delta: float = 1e-5, 
                 max_rounds: int = 300, 
                 lr: float = 0.01,
                 weighted_samples: bool = False,
                 feature_range: tuple = (0,1)
                 ):
        self.feature_range = feature_range

        super().__init__(feature_dict=feature_dict, make_binary=make_binary, pca=pca, pca_file=pca_file, seed=seed)
        self.epsilon = epsilon
        self.delta = delta
        self.max_rounds = max_rounds
        self.lr = lr
        self.weighted_samples = weighted_samples
        self.classes = np.unique(self.y_train)

    def prepare_data(self, feature_dict: dict):
        X_train, y_train, X_val, y_val, X_test, y_test = super().prepare_data(feature_dict)
        X_train, X_val, X_test = normalize_and_clip_data(X_train, X_val, X_test, feature_range=self.feature_range)
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def fit(self):  
        if len(self.classes) == 2:
            clf = DPExplainableBoostingClassifier(
                epsilon=self.epsilon, 
                delta=self.delta, 
                max_rounds=self.max_rounds, 
                learning_rate=self.lr, 
                random_state=self.seed,
                )

            if self.weighted_samples:
                sample_weight = calculate_sample_weights(self.y_train)
                self.fitted_classifier = clf.fit(self.X_train, self.y_train, sample_weight=sample_weight)
            else:
                self.fitted_classifier = clf.fit(self.X_train, self.y_train)
        
        else:   # 1-vs-all multiclass classification
            classifiers = []    # train a classifier for each class
            for cls in self.classes:
                y_train_binary = np.where(self.y_train == cls, 1, 0)    # binary label for the current class

                # the privacy budget must be distributed among the classifiers:
                # sequential composition bounds the total privacy costs of releasing
                # multiple results of dp classification on the same input data
                epsilon_multiclass = self.epsilon / len(self.classes)
                delta_multiclass = self.delta / len(self.classes)

                clf = DPExplainableBoostingClassifier(
                    epsilon=epsilon_multiclass, 
                    delta=delta_multiclass, 
                    max_rounds=self.max_rounds, 
                    learning_rate=self.lr, 
                    random_state=self.seed,
                    )

                if self.weighted_samples:
                    sample_weight = calculate_sample_weights(y_train_binary)
                    clf.fit(self.X_train, y_train_binary, sample_weight=sample_weight)

                else:
                    clf.fit(self.X_train, y_train_binary)

                classifiers.append(clf)
                self.fitted_classifier = classifiers

    def predict(self, predict_on_test: bool = False):
        X = self.X_test if predict_on_test else self.X_val
        y_true = self.y_test if predict_on_test else self.y_val

        if len(self.classes) == 2:
            y_pred = self.fitted_classifier.predict(X)
        else:
            y_pred = []
            for clf in self.fitted_classifier:
                y_pred_binary = clf.predict_proba(X)[:, 1]  # predict proba allows to access the actual prediction results to select the largest one
                y_pred.append(y_pred_binary)

            y_pred = np.array(y_pred).T      # convert the binary predictions to multiclass predictions
            y_pred = np.argmax(y_pred, axis=1)
        
        return y_true, y_pred
    

class DPGBDTTrainer(TreeTrainer):
    def __init__(self, 
                 feature_dict: dict, 
                 make_binary: bool = False, 
                 pca: float = 1.0, 
                 pca_file: str = "", 
                 seed: int = 0,
                 epsilon: float = 1.0,  # delta is set in PrivateGBDT automatically depending on the dataset size
                 num_trees: int = 100,
                 dp_method: str = "gaussian_cdp",
                 split_method: str = "totally_random",
                 sketch_type: str = "uniform",
                 undersampling: str = False,
            ):
        
        self.undersampling = undersampling
        super().__init__(feature_dict=feature_dict, make_binary=make_binary, pca=pca, pca_file=pca_file, seed=seed)
        self.epsilon = epsilon
        self.num_trees = num_trees
        self.dp_method = dp_method
        self.split_method = split_method
        self.sketch_type = sketch_type
        self.classes = np.unique(self.y_train)

    def prepare_data(self, feature_dict: dict):
        X_train, y_train, X_val, y_val, X_test, y_test = super().prepare_data(feature_dict)
        X_train, X_val, X_test = normalize_and_clip_data(X_train, X_val, X_test)
        if self.undersampling:
            X_train, y_train = undersample_data(X_train, y_train, seed=self.seed)

        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def fit(self):
        if len(self.classes) == 2:
            clf = PrivateGBDT(epsilon=self.epsilon,
                              num_trees=self.num_trees, dp_method=self.dp_method,
                              split_method=self.split_method, sketch_type=self.sketch_type)
            self.fitted_classifier = clf.fit(self.X_train, self.y_train)

        # multiclass classification
        else:
            classifiers = []    # train a classifier for each class
            for cls in self.classes:
                y_train_binary = np.where(self.y_train == cls, 1, 0)    # binary label for the current class

                # the privacy budget must be distributed among the classifiers:
                # sequential composition bounds the total privacy costs of releasing
                # multiple results of dp classification on the same input data
                epsilon_multiclass = self.epsilon / len(self.classes)

                clf = PrivateGBDT(epsilon=epsilon_multiclass, 
                                  num_trees=self.num_trees, dp_method=self.dp_method,
                                  split_method=self.split_method, sketch_type=self.sketch_type)
                clf.fit(self.X_train, y_train_binary)

                classifiers.append(clf)
                self.fitted_classifier = classifiers

    def predict(self, predict_on_test: bool = False):
        X = self.X_test if predict_on_test else self.X_val
        y_true = self.y_test if predict_on_test else self.y_val

        if len(self.classes) == 2:
            y_pred = self.fitted_classifier.predict(X)
        else:
            y_pred = []
            for clf in self.fitted_classifier:
                y_pred_binary = clf.predict_proba(X)[:, 1]  # predict proba allows to access the actual prediction results to select the largest one
                y_pred.append(y_pred_binary)

            y_pred = np.array(y_pred).T      # convert the binary predictions to multiclass predictions
            y_pred = np.argmax(y_pred, axis=1)
        
        return y_true, y_pred