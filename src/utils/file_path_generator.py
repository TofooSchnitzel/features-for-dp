from pathlib import Path
import cloudpickle
import pickle


def stats_file(config):
    path = Path(config.path.raw)/config.dataset.name/"stats"

    train_str ="trainsize=" + str(config.dataset.train_size)

    test_str = ""
    if config.dataset.name == "ham10000":
        test_str = "_testsize=" + str(config.dataset.test_size)

    file_name = train_str + test_str + ".pkl"

    return path/file_name


def feature_file(config):
    path = Path(config.path.features)/config.dataset.name/config.feature_extractor.name

    train_str = "trainsize=" + str(config.dataset.train_size)

    test_str = ""
    if config.dataset.name == "ham10000":
        test_str = "_testsize=" + str(config.dataset.test_size)

    scatter_str = ""
    if config.feature_extractor.name == "scatternet":
        scatter_str = "_J=" + str(config.feature_extractor.J) + "_L=" + str(config.feature_extractor.L) + "_m=" + str(config.feature_extractor.m)

    file_name = train_str + test_str + scatter_str + ".pkl"

    return path/file_name


def feature_file_with_pca(config):
    path = Path(config.path.features)/config.dataset.name/config.feature_extractor.name

    train_str = "trainsize=" + str(config.dataset.train_size)

    test_str = ""
    if config.dataset.name == "ham10000":
        test_str = "_testsize=" + str(config.dataset.test_size)

    scatter_str = ""
    if config.feature_extractor.name == "scatternet":
        scatter_str = "_J=" + str(config.feature_extractor.J) + "_L=" + str(config.feature_extractor.L) + "_m=" + str(config.feature_extractor.m)
    
    pca_str = "_pca=" + str(config.classifier.pca)

    file_name = train_str + test_str + scatter_str + pca_str + ".pkl"

    return path/file_name


def load_from_file(path):
    with open(path, "rb") as f:
        return pickle.load(f)
    

def save_to_file(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)