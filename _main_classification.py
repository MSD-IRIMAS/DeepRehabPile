import os

import numpy as np
import pandas as pd
import tensorflow as tf
from omegaconf import DictConfig
from sklearn.metrics import get_scorer

from deep_rehab_pile.classifiers import (
    FCN_CLASSIFIER,
    GRU_CLASSIFIER,
    LITE_MV_CLASSIFIER,
    STGCN_CLASSIFIER,
    ConvLSTM_CLASSIFIER,
    ConvTran_CLASSIFIER,
    DisjointCNN_CLASSIFIER,
    H_Inception_CLASSIFIER,
    VanTran_CLASSIFIER,
)
from deep_rehab_pile.utils import _create_directory, load_classification_data

gpus = tf.config.list_physical_devices("GPU")

if len(gpus) > 0:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def main_classification(args: DictConfig):
    """Run classification experiments.

    Main function to run classification experiments.

    Parameters
    ----------
    args: DictConfig
        The input configuration.

    Returns
    -------
    None
    """
    xtrain, ytrain, xtest, ytest, dataset_info = load_classification_data(
        dataset_name=args.dataset_name,
        root_path=os.path.join(args.root_path, "classification"),
        fold_number=args.fold_number,
    )

    length_TS = dataset_info["length_TS"]
    n_joints = dataset_info["n_joints"]
    try:
        n_dim = dataset_info["dim"]
    except KeyError:
        n_dim = dataset_info["n_dim"]

    n_classes = len(np.unique(np.concatenate((ytrain, ytest), axis=0)))

    output_dir = args.output_dir
    _create_directory(output_dir)
    output_dir_task = os.path.join(output_dir, "classification")
    _create_directory(output_dir_task)
    output_dir_dataset = os.path.join(output_dir_task, args.dataset_name)
    _create_directory(output_dir_dataset)
    output_dir_fold = os.path.join(output_dir_dataset, "fold" + str(args.fold_number))
    _create_directory(output_dir_fold)
    output_dir_model = os.path.join(output_dir_fold, args.estimator)
    _create_directory(output_dir_model)

    preds = np.zeros(shape=(len(xtest), n_classes))

    for _run in range(args.runs):
        output_dir_run = os.path.join(output_dir_model, "run" + str(_run))
        _create_directory(output_dir_run)

        _run_is_fit = False

        df_metrics = pd.DataFrame(
            columns=[
                "accuracy",
                "balanced_accuracy",
                # "f1_score",
            ]
        )
        df_time = pd.DataFrame(
            columns=[
                "training_time",
                "inference_time",
            ]
        )

        if args.estimator == "FCN":
            classifier = FCN_CLASSIFIER(
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
            classifier = H_Inception_CLASSIFIER(
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
            classifier = LITE_MV_CLASSIFIER(
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
            classifier = ConvTran_CLASSIFIER(
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
            classifier = ConvLSTM_CLASSIFIER(
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
            classifier = GRU_CLASSIFIER(
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
            classifier = VanTran_CLASSIFIER(
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
            classifier = DisjointCNN_CLASSIFIER(
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
            classifier = STGCN_CLASSIFIER(
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
            raise NotImplementedError("No classifier called " + args.estimator)

        training_time = 0.0

        if not os.path.exists(os.path.join(output_dir_run, "time.csv")):
            training_time = classifier.fit(X=xtrain, y=ytrain)
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

            preds = preds + classifier.predict_proba(X=xtest)

            ypreds, inference_time = classifier.predict(X=xtest)

            accuracy = classifier.score(X=xtest, y=ytest, metric="accuracy")
            balanced_accuracy = classifier.score(
                X=xtest, y=ytest, metric="balanced_accuracy"
            )

            df_metrics.loc[len(df_metrics)] = {
                "accuracy": accuracy,
                "balanced_accuracy": balanced_accuracy,
            }
            df_metrics.to_csv(os.path.join(output_dir_run, "metrics.csv"), index=False)

            if _run_is_fit:
                df_time.loc[len(df_time)] = {
                    "training_time": training_time,
                    "inference_time": inference_time,
                }
                df_time.to_csv(os.path.join(output_dir_run, "time.csv"), index=False)

            if args.visualize_latent_space:
                classifier.visualize_latent_space(
                    X=xtrain,
                    y=ytrain,
                    figsize=(10, 10),
                    title="Train Latent Space",
                    n_classes=n_classes,
                    pca_save_filename="pca_latent_space_train",
                    tsne_save_filename="tsne_latent_space_train",
                )

                classifier.visualize_latent_space(
                    X=xtest,
                    y=ytest,
                    figsize=(10, 10),
                    title="Test Latent Space with Ground Truth",
                    n_classes=n_classes,
                    pca_save_filename="pca_latent_space_test_ground_truth",
                    tsne_save_filename="tsne_latent_space_test_ground_truth",
                )

                classifier.visualize_latent_space(
                    X=xtest,
                    y=ypreds,
                    figsize=(10, 10),
                    title="Test Latent Space with Predictions",
                    n_classes=n_classes,
                    pca_save_filename="pca_latent_space_test_predictions",
                    tsne_save_filename="tsne_latent_space_test_predictions",
                )

    if (
        (_run_is_fit)
        or (args.force_evaluate_estimator)
        or (not os.path.exists(os.path.join(output_dir_model, "metrics_ensemble.csv")))
    ):

        preds_ensemble = preds / (1.0 * args.runs)
        preds_ensemble = np.argmax(preds_ensemble, axis=1)

        metric = get_scorer("accuracy")._score_func
        accuracy_ensemble = metric(y_true=ytest, y_pred=preds_ensemble)

        metric = get_scorer("balanced_accuracy")._score_func
        balanced_accuracy_ensemble = metric(y_true=ytest, y_pred=preds_ensemble)

        df = pd.DataFrame(
            columns=[
                "accuracy",
                "balanced_accuracy",
            ]
        )
        df.loc[len(df)] = {
            "accuracy": accuracy_ensemble,
            "balanced_accuracy": balanced_accuracy_ensemble,
        }

        df.to_csv(os.path.join(output_dir_model, "metrics_ensemble.csv"), index=False)
