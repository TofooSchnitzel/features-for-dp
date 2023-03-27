from classification.cnn import CNNTrainer, DPCNNTrainer
from classification.linear import DPLinearTrainer
from classification.trees import TabPFNTrainer, EBMTrainer, GBDTTrainer, LinearTrainer, DPEBMTrainer, DPGBDTTrainer
from utils.file_path_generator import feature_file_with_pca


def prepare_training(config, feature_dict: dict, seed: int = 0):
    make_binary = True if (config.dataset.name == "pppp" and len(config.dataset.classes) == 2) else False

    if "pca" in config.classifier and config.classifier.pca < 1.0:
        pca_file = feature_file_with_pca(config)
        pca = config.classifier.pca
    else:
        pca_file = ""
        pca=1.0


    match config.privacy_setting:
        case "baseline":
            match config.classifier.name:
                case "tabpfn":
                    trainer = TabPFNTrainer(
                        feature_dict=feature_dict, 
                        make_binary=make_binary, 
                        pca=pca, 
                        pca_file=pca_file, 
                        seed=seed
                        )
                case "ebm":
                    trainer = EBMTrainer(
                        feature_dict=feature_dict, 
                        make_binary=make_binary, 
                        max_rounds=config.classifier.max_rounds,
                        pca=pca, 
                        pca_file=pca_file, 
                        seed=seed
                        )
                case "gbdt":
                    trainer = GBDTTrainer(
                        feature_dict=feature_dict, 
                        make_binary=make_binary, 
                        n_estimators=config.classifier.n_estimators, 
                        pca=pca, 
                        pca_file=pca_file, 
                        seed=seed
                        )
                case "linear":
                    trainer = LinearTrainer(
                        feature_dict=feature_dict, 
                        make_binary=make_binary, 
                        pca=pca, 
                        pca_file=pca_file, 
                        seed=seed
                        )

                case "cnn":
                    trainer = CNNTrainer(
                        feature_dict=feature_dict, 
                        make_binary=make_binary,
                        batch_size=config.classifier.batch_size,
                        epochs=config.classifier.epochs,
                        lr=config.classifier.lr,
                        scale_norm=config.classifier.scale_norm,
                        log_wandb=config.log_wandb,
                        seed=seed
                        )
                
        case "dp":
            match config.classifier.name:
                case "dp_ebm":
                    trainer = DPEBMTrainer(
                        feature_dict=feature_dict,
                        make_binary=make_binary,
                        pca=pca,
                        pca_file=pca_file,
                        seed=seed,
                        epsilon=config.classifier.epsilon,
                        delta=config.classifier.delta,
                        max_rounds=config.classifier.max_rounds,
                        lr=config.classifier.lr,
                        weighted_samples=config.classifier.weighted_samples,
                    )
                case "dp_gbdt":
                    trainer = DPGBDTTrainer(
                        feature_dict=feature_dict,
                        make_binary=make_binary,
                        pca=pca,
                        pca_file=pca_file,
                        seed=seed,
                        epsilon=config.classifier.epsilon,  # delta is set in PrivateGBDT automatically depending on the dataset size
                        num_trees=config.classifier.num_trees,
                        dp_method=config.classifier.dp_method,
                        split_method=config.classifier.split_method,
                        sketch_type=config.classifier.sketch_type,
                        undersampling=config.classifier.undersampling,
                    )
                case "dp_cnn":
                    trainer = DPCNNTrainer(
                        feature_dict,
                        make_binary=make_binary,
                        batch_size=config.classifier.batch_size,
                        epochs=config.classifier.epochs,
                        lr=config.classifier.lr,
                        scale_norm=config.classifier.scale_norm,
                        epsilon=config.classifier.epsilon,
                        delta=config.classifier.delta,
                        max_grad_norm=config.classifier.max_grad_norm,
                        max_physical_batch_size=config.classifier.max_physical_batch_size,
                        log_wandb=config.log_wandb,
                        seed=seed,
                        )
                case "dp_linear":
                    trainer = DPLinearTrainer(
                        feature_dict, 
                        make_binary=make_binary,
                        batch_size=config.classifier.batch_size,
                        epochs=config.classifier.epochs,
                        lr=config.classifier.lr,
                        optimizer_name=config.classifier.optimizer_name,
                        epsilon=config.classifier.epsilon,
                        delta=config.classifier.delta,
                        max_grad_norm=config.classifier.max_grad_norm,
                        max_physical_batch_size=config.classifier.max_physical_batch_size,
                        log_wandb=config.log_wandb,
                        seed=seed,
                    )
    return trainer