def check_config_is_valid(config):
    if config.dataset.name == "pppp":   # using a model pre-trained on imagenet should also run, but probably not the best choice
        assert config.feature_extractor.name in ["radimagenet", "scatternet"]

    if config.dataset.name == "ham10000": 
        assert config.feature_extractor.name in ["imagenet", "scatternet"]

    if config.feature_extractor.name == "scatternet":
        assert config.feature_extractor.J < 8   # the smallest dimension of the image, 224, needs to be larger than 2^J

    if config.classifier == "tabpfn":       # assert number of features is max 100
        if not (config.feature_extractor.name == "scatternet" and config.feature_extractor.m == 1 and config.feature_extractor.J == 7):     # only 57 features
            assert "pca" in config.classifier   

        assert not (config.feature_extractor.name == "scatternet" and config.feature_extractor.J == 1)   # 101 features

    if config.privacy_setting == "dp":
        assert config.classifier.name in ["dp_cnn", "dp_ebm", "dp_gbdt", "dp_linear"]
    else:
        assert config.classifier.name in ["cnn", "ebm", "gbdt", "tabpfn", "linear"]

    if config.feature_extractor.name in ["imagenet", "radimagenet"]:
        assert config.classifier.name not in ["cnn", "dp_cnn"]


def project_name(config):
    ds = config.dataset.name
    if config.dataset.name == "pppp":
        ds += str(len(config.dataset.classes))

    evaluation = "evaluation_" if ("evaluation" in config and config.evaluation == True) else ""
    project_name = evaluation + str(config.privacy_setting) + "_" + ds
    
    return project_name
