"""Base deep regressor."""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import get_scorer


class BASE_REGRESSOR:
    """Define base deep regressor.

    Parameters
    ----------
    output_dir: str,
        The output directory.
    best_file_name: str,
        The name of the best model to save.
    init_file_name: str,
        The name of the init model to save.
    length_TS: int,
        The length of input skeleton sequence.
    n_joints: int,
        The number of joints in the skeleton.
    n_dim: int,
        The number of dimensions per joint.
    batch_size: int, default = 64,
        The batch size used for training.
    epochs: int, default = 1500,
        The number of epochs used to train the regressor.

    Returns
    -------
    None
    """

    def __init__(
        self,
        output_dir: str,
        best_file_name: str,
        init_file_name: str,
        length_TS: int,
        n_joints: int,
        n_dim: int,
        batch_size: int = 64,
        epochs: int = 1500,
    ) -> None:
        self.output_dir = output_dir
        self.best_file_name = best_file_name
        self.init_file_name = init_file_name
        self.length_TS = length_TS
        self.n_joints = n_joints
        self.n_dim = n_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.regressor = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Train the regressor.

        Parameters
        ----------
        X: np.ndarray, shape = (n_samples, n_timepoints, n_channels),
            The input samples.
        y: np.ndarray, shape = (n_samples),
            The score labels.

        Returns
        -------
        delta_time: float,
            The training time.
        """
        start_time = time.time()
        self.regressor.fit(X=X, y=y)
        delta_time = time.time() - start_time
        tf.keras.backend.clear_session()

        return delta_time

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Predict on new samples.

        Parameters
        ----------
        X: np.ndarray, shape = (n_samples, n_timepoints, n_channels),
            The input samples.

        Returns
        -------
        preds: np.ndarray, shape = (n_samples),
            The predicted score labels.
        delta_time: float,
            The inference time.
        """
        start_time = time.time()
        model = tf.keras.models.load_model(
            self.output_dir + self.best_file_name + ".keras", compile=False
        )

        X = np.swapaxes(X, axis1=1, axis2=2)

        preds = model.predict(X).reshape((len(X),))

        preds[preds > 1.0] = 1.0
        preds[preds < 0.0] = 0.0

        delta_time = time.time() - start_time

        return preds, delta_time

    def score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        max_score_value: float,
        metric: str = "neg_mean_squared_error",
    ) -> float:
        """
        Get score following some metric.

        Parameters
        ----------
        X: np.ndarray, shape = (n_samples, n_timepoints, n_channels),
            The input samples.
        y: np.ndarray, shape = (n_samples),
            The score labels.
        max_score_value: float,
            The maximum value for scores in the regression dataset.
        metric: str, default = "neg_root_mean_squared_error",
            The metric to be used, taken from sklearn metrics list.
            Possibilities: neg_root_mean_squared_error, neg_mean_absolute_error,
                           neg_mean_absolute_percentage_error.

        Return
        ------
        score : float,
            The output metric.
        """
        metric_func = get_scorer(metric)._score_func
        ypred, _ = self.predict(X)

        ypred_re_scaled = ypred * (max_score_value * 1.0)
        y_re_scaled = y * (max_score_value * 1.0)

        if metric == "neg_mean_absolute_percentage_error":
            return metric_func(y_true=y_re_scaled + 1.0, y_pred=ypred_re_scaled + 1.0)
        else:
            return metric_func(y_true=y_re_scaled, y_pred=ypred_re_scaled)

    def visualize_latent_space(
        self,
        X: np.ndarray,
        y: np.ndarray,
        max_score_value: float,
        figsize: tuple[int, int],
        title: str,
        pca_save_filename: str,
        tsne_save_filename: str,
    ) -> None:
        """
        Visualize latent space features of pretrained model.

        Parameters
        ----------
        X: np.ndarray, shape = (n_samples, n_timepoints, n_channels),
            The input samples.
        y: np.ndarray, shape = (n_samples),
            The score labels.
        max_score_value: float,
            The maximum value for scores in the regression dataset.
        figsize: tuple[int,int],
            The size of the figure.
        title: str,
            The title of the figure.
        pca_save_filename: str,
            The name of the PCA saved figure, without the .pdf extension.
        tsne_save_filename: str,
            The name of the t-SNE saved figure, without the .pdf extension.

        Returns
        -------
        None
        """
        X = np.swapaxes(X, axis1=1, axis2=2)

        model = tf.keras.models.load_model(
            self.output_dir + self.best_file_name + ".keras", compile=False
        )

        new_input = model.input
        new_output = model.layers[-2].output

        new_model = tf.keras.models.Model(inputs=new_input, outputs=new_output)

        X_latent = new_model.predict(X)

        pca = PCA(n_components=2, random_state=42)
        X_latent_2d_pca = pca.fit_transform(X_latent)

        perplexity = 30.0
        if len(X) <= perplexity:
            perplexity = len(X) / 2
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        X_latent_2d_tsne = tsne.fit_transform(X_latent)

        fig = plt.figure(figsize=figsize)

        ax = fig.add_subplot(111)

        scatter = ax.scatter(
            X_latent_2d_pca[:, 0],
            X_latent_2d_pca[:, 1],
            c=y * (max_score_value * 1.0),
            s=200,
            cmap="jet",
        )

        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("Regression Values")

        ax.set_title(title + " - PCA", fontsize=20)

        fig.savefig(
            os.path.join(self.output_dir, pca_save_filename + ".pdf"),
            bbox_inches="tight",
        )

        fig.clf()

        fig = plt.figure(figsize=figsize)

        ax = fig.add_subplot(111)

        scatter = ax.scatter(
            X_latent_2d_tsne[:, 0],
            X_latent_2d_tsne[:, 1],
            c=y * (max_score_value * 1.0),
            s=200,
            cmap="jet",
        )

        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("Regression Values")

        ax.set_title(title + " - tSNE", fontsize=20)

        fig.savefig(
            os.path.join(self.output_dir, tsne_save_filename + ".pdf"),
            bbox_inches="tight",
        )
