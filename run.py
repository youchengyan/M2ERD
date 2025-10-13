# coding=utf-8
import os
from datetime import datetime
from ray import tune

# grid search - weibo
from AvgMetricCallback import AvgMetricCallback
from train import TrainALL


def custom_dirname_creator(trial):
    return f"exp_{trial.trial_id}"


local_dir_root = 'root dir'
repeat_times = 10
max_concurrent = 1
avg_metric = AvgMetricCallback()

analysis = tune.run(
    TrainALL,
    callbacks=[avg_metric],
    metric="best_test_accuracy",
    mode="max",
    name="weibo-experiment",
    trial_dirname_creator=custom_dirname_creator,
    local_dir=os.path.join(local_dir_root, '{}'.format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))),
    resources_per_trial={"cpu": 1, "gpu": 1 / max_concurrent},
    stop={"training_iteration": 1},
    num_samples=repeat_times,
    config={
        # data
        "dataset_name": tune.grid_search(["weibo"]),
        "max_sen_len": tune.grid_search([80]),

        # Network
        "ablation": tune.grid_search(
            ["bert+dct+vgg+fusion"]),  # 'bert', 'vgg', 'dct', 'bert+vgg+fusion', 'bert+dct+vgg+concat', 'bert+vgg+concat',

        "bert_model_name": tune.grid_search(["bert-base-uncased"]),
        "kernel_sizes": tune.grid_search([[3, 3, 3]]),
        "num_channels": tune.grid_search([[32, 64, 128]]),
        "num_layers": tune.grid_search([2]),
        "num_heads": tune.grid_search([4]),
        "dropout": tune.grid_search([0.5]),
        "drop_and_BN": tune.grid_search(['drop-BN']),  # 'drop-BN', 'BN-drop', 'BN-only', 'drop-only', 'none'
        "FREEZE_BERT": tune.grid_search([False]),
        "FREEZE_VGG": tune.grid_search([False]),
        "model_dim": tune.grid_search([256]),
        "init_method": tune.grid_search(['default']),

        # optimizer
        "optimizer_name": tune.grid_search(["AdaBelief"]),
        "learning_rate": tune.grid_search([0.0001]),
        "bert_learning_rate": tune.loguniform(1e-5, 1e-2),
        "vgg_learning_rate": tune.loguniform(1e-5, 1e-2),
        "dtcconv_learning_rate": tune.loguniform(1e-5, 1e-2),
        "fusion_learning_rate": tune.loguniform(1e-5, 1e-2),
        "linear_learning_rate": tune.loguniform(1e-5, 1e-2),
        "classifier_learning_rate": tune.loguniform(1e-5, 1e-2),

        "momentum": tune.grid_search([0.9]),
        "weight_decay": tune.grid_search([0.15]),
        "seed": tune.grid_search([43]),

        "early_stopping_patience": 10,

        # training
        "epochs": tune.grid_search([60]),
        "train_bs": tune.grid_search([2]), # batch_size
        "test_bs": tune.grid_search([2]),
        "confidence": tune.grid_search([0.7, 0.8, 0.9])
    },
)

print("Best config is:", analysis.get_best_config(metric="best_test_accuracy", mode="max"))
