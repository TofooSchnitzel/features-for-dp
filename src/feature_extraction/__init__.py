from feature_extraction import scatternet
from feature_extraction import pretrained_cnn


def select_feature_extractor(config):
    match config.feature_extractor.name:
        case "scatternet":
            feature_extractor = scatternet.FeatureExtractor(
                J=config.feature_extractor.J, 
                L=config.feature_extractor.L, 
                m=config.feature_extractor.m
                )
        case "radimagenet":
            feature_extractor = pretrained_cnn.FeatureExtractor(modeltype="radimagenet")
        case "imagenet":
            feature_extractor = pretrained_cnn.FeatureExtractor(modeltype="imagenet")
        case other:
            raise NotImplementedError(f"Not implemented for dataset {config.feature_extractor.name}")

    return feature_extractor