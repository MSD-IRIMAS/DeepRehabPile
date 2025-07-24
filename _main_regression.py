import os

import numpy as np
import pandas as pd
import tensorflow as tf
from omegaconf import DictConfig
from sklearn.metrics import get_scorer

from deep_rehab_pile.regressors import (
    FCN_REGRESSOR,
    GRU_REGRESSOR,
    LITE_MV_REGRESSOR,
    STGCN_REGRESSOR,
    ConvLSTM_REGRESSOR,
    ConvTran_REGRESSOR,
    DisjointCNN_REGRESSOR,
    H_Inception_REGRESSOR,
    VanTran_REGRESSOR,
)
from deep_rehab_pile.utils import _create_directory, load_regression_data

gpus = tf.config.list_physical_devices("GPU")
if len(gpus) > 0:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def main_regression(args: DictConfig):
    """Run regression experiments.

    Main function to run regression experiments.

    Parameters
    ----------
    args: DictConfig
        The input configuration.

    Returns
    -------
    None
    """
    xtrain, ytrain, xtest, ytest, dataset_info = load_regression_data(
        dataset_name=args.dataset_name,
        root_path=os.path.join(args.root_path, "regression"),
        fold_number=args.fold_number,
    )

    length_TS = dataset_info["length_TS"]
    n_joints = dataset_info["n_joints"]
    try:
        n_dim = dataset_info["dim"]
    except KeyError:
        n_dim = dataset_info["n_dim"]

    output_dir = args.output_dir
    _create_directory(output_dir)
    output_dir_task = os.path.join(output_dir, "regression")
    _create_directory(output_dir_task)
    output_dir_dataset = os.path.join(output_dir_task, args.dataset_name)
    _create_directory(output_dir_dataset)
    output_dir_fold = os.path.join(output_dir_dataset, "fold" + str(args.fold_number))
    _create_directory(output_dir_fold)
    output_dir_model = os.path.join(output_dir_fold, args.estimator)
    _create_directory(output_dir_model)

    preds = np.zeros(shape=(len(xtest)))

    for _run in range(args.runs):
        output_dir_run = os.path.join(output_dir_model, "run" + str(_run))
        _create_directory(output_dir_run)

        _run_is_fit = False

        df_metrics = pd.DataFrame(
            columns=[
                "rmse",
                "mae",
                "mape",
            ]
        )

        df_time = pd.DataFrame(columns=["training_time", "inference_time"])

        if args.estimator == "FCN":
            regressor = FCN_REGRESSOR(
                output_dir=output_dir_run + "/",
                best_file_name="best_model",
                init_file_name="init_model",
                length_TS=length_TS,
                n_joints=n_joints,
                n_dim=n_dim,
                epochs=args.epochs,
                batch_size=args.batch_size,
                **args.estimator_params["FCN"],
            )
        elif args.estimator == "H_Inception":
            regressor = H_Inception_REGRESSOR(
                output_dir=output_dir_run + "/",
                best_file_name="best_model",
                init_file_name="init_model",
                length_TS=length_TS,
                n_joints=n_joints,
                n_dim=n_dim,
                epochs=args.epochs,
                batch_size=args.batch_size,
                **args.estimator_params["H_Inception"],
            )
        elif args.estimator == "LITEMV":
            regressor = LITE_MV_REGRESSOR(
                output_dir=output_dir_run + "/",
                best_file_name="best_model",
                init_file_name="init_model",
                length_TS=length_TS,
                n_joints=n_joints,
                n_dim=n_dim,
                epochs=args.epochs,
                batch_size=args.batch_size,
                **args.estimator_params["LITEMV"],
            )
        elif args.estimator == "ConvTran":
            regressor = ConvTran_REGRESSOR(
                output_dir=output_dir_run + "/",
                best_file_name="best_model",
                init_file_name="init_model",
                length_TS=length_TS,
                n_joints=n_joints,
                n_dim=n_dim,
                epochs=args.epochs,
                batch_size=args.batch_size,
                **args.estimator_params["ConvTran"],
            )
        elif args.estimator == "ConvLSTM":
            regressor = ConvLSTM_REGRESSOR(
                output_dir=output_dir_run + "/",
                best_file_name="best_model",
                init_file_name="init_model",
                length_TS=length_TS,
                n_joints=n_joints,
                n_dim=n_dim,
                epochs=args.epochs,
                batch_size=args.batch_size,
                **args.estimator_params["ConvLSTM"],
            )
        elif args.estimator == "GRU":
            regressor = GRU_REGRESSOR(
                output_dir=output_dir_run + "/",
                best_file_name="best_model",
                init_file_name="init_model",
                length_TS=length_TS,
                n_joints=n_joints,
                n_dim=n_dim,
                epochs=args.epochs,
                batch_size=args.batch_size,
                **args.estimator_params["GRU"],
            )
        elif args.estimator == "VanTran":
            regressor = VanTran_REGRESSOR(
                output_dir=output_dir_run + "/",
                best_file_name="best_model",
                init_file_name="init_model",
                length_TS=length_TS,
                n_joints=n_joints,
                n_dim=n_dim,
                epochs=args.epochs,
                batch_size=args.batch_size,
                **args.estimator_params["VanTran"],
            )
        elif args.estimator == "DisjointCNN":
            regressor = DisjointCNN_REGRESSOR(
                output_dir=output_dir_run + "/",
                best_file_name="best_model",
                init_file_name="init_model",
                length_TS=length_TS,
                n_joints=n_joints,
                n_dim=n_dim,
                epochs=args.epochs,
                batch_size=args.batch_size,
                **args.estimator_params["DisjointCNN"],
            )
        elif args.estimator == "STGCN":
            regressor = STGCN_REGRESSOR(
                output_dir=output_dir_run + "/",
                best_file_name="best_model",
                init_file_name="init_model",
                length_TS=length_TS,
                n_joints=n_joints,
                n_dim=n_dim,
                kinematic_tree=dataset_info["kinematic_tree"],
                epochs=args.epochs,
                batch_size=args.batch_size,
                **args.estimator_params["STGCN"],
            )
        else:
            raise NotImplementedError("No regressor called " + args.regressor)

        training_time = 0.0

        if args.train_estimator:
            if not os.path.exists(os.path.join(output_dir_run, "time.csv")):
                training_time = regressor.fit(X=xtrain, y=ytrain)
                _run_is_fit = True
        else:
            temp_df_time = pd.read_csv(os.path.join(output_dir_run, "time.csv"))
            training_time = temp_df_time.iloc[0]["training_time"]
            inference_time = temp_df_time.iloc[0]["inference_time"]

            assert isinstance(training_time, float)
            assert isinstance(inference_time, float)

        if (_run_is_fit) or (
            not os.path.exists(os.path.join(output_dir_run, "metrics.csv"))
            or (args.force_evaluate_estimator)
        ):

            preds_run, inference_time = regressor.predict(X=xtest)

            preds = preds + preds_run.reshape((-1,))

            max_score_value = dataset_info["max_score_value"]

            rmse = regressor.score(
                X=xtest,
                y=ytest,
                metric="neg_root_mean_squared_error",
                max_score_value=max_score_value,
            )
            mae = regressor.score(
                X=xtest,
                y=ytest,
                metric="neg_mean_absolute_error",
                max_score_value=max_score_value,
            )
            mape = regressor.score(
                X=xtest,
                y=ytest,
                metric="neg_mean_absolute_percentage_error",
                max_score_value=max_score_value,
            )

            df_metrics.loc[len(df_metrics)] = {
                "rmse": rmse,
                "mae": mae,
                "mape": mape,
            }
            df_metrics.to_csv(os.path.join(output_dir_run, "metrics.csv"), index=False)

            if _run_is_fit:
                df_time.loc[len(df_time)] = {
                    "training_time": training_time,
                    "inference_time": inference_time,
                }
                df_time.to_csv(os.path.join(output_dir_run, "time.csv"), index=False)

            # regressor.visualize_latent_space(
            #     X=xtrain,
            #     y=ytrain,
            #     figsize=(10, 10),
            #     title="Train Latent Space",
            #     pca_save_filename="pca_latent_space_train",
            #     tsne_save_filename="tsne_latent_space_train",
            #     max_score_value=max_score_value,
            # )

            # regressor.visualize_latent_space(
            #     X=xtest,
            #     y=ytest,
            #     figsize=(10, 10),
            #     title="Test Latent Space with Ground Truth",
            #     pca_save_filename="pca_latent_space_test_ground_truth",
            #     tsne_save_filename="tsne_latent_space_test_ground_truth",
            #     max_score_value=max_score_value,
            # )

            # regressor.visualize_latent_space(
            #     X=xtest,
            #     y=preds_run,
            #     figsize=(10, 10),
            #     title="Test Latent Space with Predictions",
            #     pca_save_filename="pca_latent_space_test_predictions",
            #     tsne_save_filename="tsne_latent_space_test_predictions",
            #     max_score_value=max_score_value,
            # )

    if (
        (_run_is_fit)
        or (args.force_evaluate_estimator)
        or (not os.path.exists(os.path.join(output_dir_model, "metrics_ensemble.csv")))
    ):

        preds_ensemble = preds / (1.0 * args.runs)

        metric = get_scorer("neg_root_mean_squared_error")._score_func
        rmse_ensemble = metric(
            y_true=ytest * (max_score_value * 1.0),
            y_pred=preds_ensemble * (max_score_value * 1.0),
        )

        metric = get_scorer("neg_mean_absolute_error")._score_func
        mae_ensemble = metric(
            y_true=ytest * (max_score_value * 1.0),
            y_pred=preds_ensemble * (max_score_value * 1.0),
        )

        metric = get_scorer("neg_mean_absolute_percentage_error")._score_func
        mape_ensemble = metric(
            y_true=ytest * (max_score_value * 1.0) + 1.0,
            y_pred=preds_ensemble * (max_score_value * 1.0) + 1.0,
        )

        df = pd.DataFrame(
            columns=[
                "rmse",
                "mae",
                "mape",
            ]
        )
        df.loc[len(df)] = {
            "rmse": rmse_ensemble,
            "mae": mae_ensemble,
            "mape": mape_ensemble,
        }

        df.to_csv(os.path.join(output_dir_model, "metrics_ensemble.csv"), index=False)
