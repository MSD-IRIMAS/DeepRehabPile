"""Utils functions."""

__all__ = [
    "_create_directory",
    "_normalize_skeletons",
    "load_regression_data",
    "load_classification_data",
]

import json
import os
import re
from typing import Tuple

import numpy as np
from sklearn.preprocessing import LabelEncoder


def _create_directory(directory_path: str) -> None:
    """
    Create a directory if it doesn't exist.

    Parameters
    ----------
    directory_path : str
       The path of the directory to create.
    """
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
        except FileExistsError:
            raise FileExistsError("Already exists.")


def _normalize_skeletons(
    X: np.ndarray,
    min_max_list: list = None,
):
    dim = int(X.shape[2])
    n_X = np.zeros(shape=X.shape)

    if min_max_list is None:
        min_max_list = []
        for d in range(dim):
            min_ = np.min(X[:, :, d, :])
            max_ = np.max(X[:, :, d, :])
            min_max_list.append((min_, max_))

    for d in range(dim):
        n_X[:, :, d, :] = (X[:, :, d, :] - min_max_list[d][0]) / (
            1.0 * (min_max_list[d][1] - min_max_list[d][0])
        )

    return n_X, min_max_list


def load_regression_data(
    dataset_name: str,
    root_path: str,
    fold_number: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load the chosen regression dataset.

    Parameters
    ----------
    dataset_name: str
        The name of the regression dataset chosen to load.
    root_path: str
        The directory containing all datasets.
    fold_number: int
        The fold number.

    Returns
    -------
    xtrain: np.ndarray
        The training sequences of shape (n_cases, n_channels, n_timepoints).
    ytrain: np.ndarray
        The labels of the training samples of shape (n_cases,).
    xtest: np.ndarray
        The testing sequences of shape (n_cases, n_channels, n_timepoints).
    ytest: np.ndarray
        The labels of the testing samples of shape (n_cases,).
    dataset_info: dict
        The dataset information in dictionary format.
    """
    dataset_path = os.path.join(root_path, dataset_name, "fold" + str(fold_number))

    # f = open(os.path.join(root_path, dataset_name, "info.json"))
    # dataset_info = json.load(f)

    with open(os.path.join(root_path, dataset_name, "info.json")) as f:
        content_info_datset = f.read()

    content_info_datset = re.sub(r",\s*([\]}])", r"\1", content_info_datset)
    content_info_datset = re.sub(r",\s*([\]}])", r"\1", content_info_datset)

    dataset_info = json.loads(content_info_datset)

    length_TS = dataset_info["length_TS"]
    n_joints = dataset_info["n_joints"]
    try:
        dim = dataset_info["dim"]
    except KeyError:
        dim = dataset_info["n_dim"]

    try:
        max_score_value = dataset_info["max_score_value"]
    except KeyError:
        max_score_value = 100.0

    xtrain = np.load(
        os.path.join(dataset_path, "x_train_fold" + str(fold_number) + ".npy"),
        allow_pickle=True,
    )
    ytrain = np.load(
        os.path.join(dataset_path, "y_train_fold" + str(fold_number) + ".npy"),
        allow_pickle=True,
    )
    xtest = np.load(
        os.path.join(dataset_path, "x_test_fold" + str(fold_number) + ".npy"),
        allow_pickle=True,
    )
    ytest = np.load(
        os.path.join(dataset_path, "y_test_fold" + str(fold_number) + ".npy"),
        allow_pickle=True,
    )

    xtrain = np.array(xtrain.tolist())
    ytrain = np.array(ytrain.tolist())
    xtest = np.array(xtest.tolist())
    ytest = np.array(ytest.tolist())

    xtrain = np.reshape(xtrain, (len(xtrain), n_joints, dim, length_TS))
    xtest = np.reshape(xtest, (len(xtest), n_joints, dim, length_TS))

    xtrain, min_max_list = _normalize_skeletons(X=xtrain, min_max_list=None)

    xtest, _ = _normalize_skeletons(X=xtest, min_max_list=min_max_list)

    xtrain = np.reshape(xtrain, (len(xtrain), n_joints * dim, length_TS))
    xtest = np.reshape(xtest, (len(xtest), n_joints * dim, length_TS))

    ytrain = ytrain / (max_score_value * 1.0)
    ytest = ytest / (max_score_value * 1.0)

    f.close()

    return (
        xtrain,
        np.reshape(ytrain, (len(ytrain),)),
        xtest,
        np.reshape(ytest, (len(ytest),)),
        dataset_info,
    )


def load_classification_data(
    dataset_name: str, root_path: str, fold_number: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load the chosen classification dataset.

    Parameters
    ----------
    dataset_name: str
        The name of the classification dataset chosen to load.
    root_path: str
        The directory containing all datasets.
    fold_number: int
        The fold number.

    Returns
    -------
    xtrain: np.ndarray
        The training sequences of shape (n_cases, n_channels, n_timepoints).
    ytrain: np.ndarray
        The labels of the training samples of shape (n_cases,).
    xtest: np.ndarray
        The testing sequences of shape (n_cases, n_channels, n_timepoints).
    ytest: np.ndarray
        The labels of the testing samples of shape (n_cases,).
    dataset_info: dict
        The dataset information in dictionary format.
    """
    dataset_path = os.path.join(root_path, dataset_name, "fold" + str(fold_number))

    # f = open(os.path.join(root_path, dataset_name, "info.json"))
    # dataset_info = json.load(f)

    with open(os.path.join(root_path, dataset_name, "info.json")) as f:
        content_info_datset = f.read()

    content_info_datset = re.sub(r",\s*([\]}])", r"\1", content_info_datset)
    content_info_datset = re.sub(r",\s*([\]}])", r"\1", content_info_datset)

    dataset_info = json.loads(content_info_datset)

    length_TS = dataset_info["length_TS"]
    n_joints = dataset_info["n_joints"]
    try:
        dim = dataset_info["dim"]
    except KeyError:
        dim = dataset_info["n_dim"]

    xtrain = np.load(
        os.path.join(dataset_path, "x_train_fold" + str(fold_number) + ".npy"),
        allow_pickle=True,
    )
    ytrain = np.load(
        os.path.join(dataset_path, "y_train_fold" + str(fold_number) + ".npy"),
        allow_pickle=True,
    )
    xtest = np.load(
        os.path.join(dataset_path, "x_test_fold" + str(fold_number) + ".npy"),
        allow_pickle=True,
    )
    ytest = np.load(
        os.path.join(dataset_path, "y_test_fold" + str(fold_number) + ".npy"),
        allow_pickle=True,
    )

    xtrain = np.array(xtrain.tolist())
    ytrain = np.array(ytrain.tolist())
    xtest = np.array(xtest.tolist())
    ytest = np.array(ytest.tolist())

    xtrain = np.reshape(xtrain, (len(xtrain), n_joints, dim, length_TS))
    xtest = np.reshape(xtest, (len(xtest), n_joints, dim, length_TS))

    xtrain, min_max_list = _normalize_skeletons(X=xtrain, min_max_list=None)

    xtest, _ = _normalize_skeletons(X=xtest, min_max_list=min_max_list)

    xtrain = np.reshape(xtrain, (len(xtrain), n_joints * dim, length_TS))
    xtest = np.reshape(xtest, (len(xtest), n_joints * dim, length_TS))

    le = LabelEncoder()
    ytrain = le.fit_transform(ytrain)
    ytest = le.transform(ytest)

    f.close()

    return (
        xtrain,
        np.reshape(ytrain, (len(ytrain),)),
        xtest,
        np.reshape(ytest, (len(ytest),)),
        dataset_info,
    )
