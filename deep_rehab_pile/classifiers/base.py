"""Base deep classifier."""

__all__ = ["BASE_CLASSIFIER"]

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.markers import MarkerStyle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import get_scorer


class BASE_CLASSIFIER:
    """Define the base class for deep classifiers.

    The base deep classifier implements the common functions between
    all models.

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
        The number of epochs used to train the classifier.

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
        self.classifier = None
        self.n_classes = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Train the classifier.

        Parameters
        ----------
        X: np.ndarray, shape = (n_samples, n_channels, n_timepoints),
            The input samples.
        y: np.ndarray, shape = (n_samples),
            The class labels.

        Returns
        -------
        delta_time: float,
            The training time.
        """
        start_time = time.time()
        self.classifier.fit(X=X, y=y)
        delta_time = time.time() - start_time

        self.n_classes = self.classifier.n_classes_

        tf.keras.backend.clear_session()

        return delta_time

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Predict on new samples.

        Parameters
        ----------
        X: np.ndarray, shape = (n_samples, n_channels, n_timepoints),
            The input samples.

        Returns
        -------
        preds: np.ndarray, shape = (n_samples),
            The predicted class labels.
        delta_time: float,
            The inference time.
        """
        start_time = time.time()
        preds = np.argmax(self.predict_proba(X=X), axis=1)
        delta_time = time.time() - start_time

        return preds, delta_time

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict proba on new samples.

        Parameters
        ----------
        X: np.ndarray, shape = (n_samples, n_channels, n_timepoints),
            The input samples.

        Returns
        -------
        preds_proba: np.ndarray, shape = (n_samples, n_classes),
            The predicted class labels.
        """
        model = tf.keras.models.load_model(
            self.output_dir + self.best_file_name + ".keras", compile=False
        )

        X = np.swapaxes(X, axis1=1, axis2=2)

        return model.predict(X)

    def score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metric: str = "accuracy",
        n_classes: int = None,
    ) -> float:
        """
        Get score following some metric.

        Parameters
        ----------
        X: np.ndarray, shape = (n_samples, n_channels, n_timepoints),
            The input samples.
        y: np.ndarray, shape = (n_samples),
            The class labels.
        metric: str, default = "accuracy",
            The metric to be used, taken from sklearn metrics list.
            Possibilities: accuracy, balanced_accuracy, f1.
        n_classes: int, default = None,
            The number of classes, used for evaluation when fit is not
            called, if fit is called this is assigned internally and can
            be ignored.

        Return
        ------
        score : float,
            The output metric.
        """
        if self.n_classes is None:
            self.n_classes = n_classes

        # if metric == "f1" and self.n_classes > 2:
        if metric == "f1":
            metric = get_scorer(metric)._score_func
            ypred, _ = self.predict(X)

            return metric(
                y_true=y,
                y_pred=ypred,
                average="macro",
                labels=np.arange(self.n_classes),
            )
        # elif metric == "f1" and self.n_classes <= 2:
        #     metric = get_scorer(metric)._score_func
        #     ypred, _ = self.predict(X)

        #     return metric(y_true=y, y_pred=ypred, labels=np.arange(self.n_classes))
        else:
            metric = get_scorer(metric)._score_func
            ypred, _ = self.predict(X)

            return metric(y_true=y, y_pred=ypred)

    def visualize_latent_space(
        self,
        X: np.ndarray,
        y: np.ndarray,
        figsize: tuple[int, int],
        title: str,
        n_classes: int,
        pca_save_filename: str,
        tsne_save_filename: str,
    ):
        """
        Visualize latent space features of pretrained model.

        Parameters
        ----------
        X: np.ndarray, shape = (n_samples, n_timepoints, n_channels),
            The input samples.
        y: np.ndarray, shape = (n_samples),
            The score labels.
        figsize: tuple[int,int],
            The size of the figure.
        title: str,
            The title of the figure.
        n_classes: int,
            The total number of possible classes.
        pca_save_filename: str,
            The name of the PCA saved figure, without the .pdf extension.
        tsne_save_filename: str,
            The name of the t-SNE saved figure, without the .pdf extension.

        Returns
        -------
        None
        """
        if os.path.exists(
            os.path.join(self.output_dir, pca_save_filename + ".pdf")
        ) and os.path.exists(
            os.path.join(self.output_dir, tsne_save_filename + ".pdf")
        ):
            return

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

        cmap = plt.get_cmap("tab20")
        # colors = [cmap(random.random()) for _ in range(n_classes)]
        colors = [cmap(i / n_classes) for i in range(n_classes)]

        rng = np.random.default_rng(seed=42)
        all_markers = MarkerStyle.filled_markers
        markers = rng.choice(all_markers, size=n_classes, replace=False)

        fig = plt.figure(figsize=figsize)

        ax = fig.add_subplot(111)

        for c in range(n_classes):

            ax.scatter(
                X_latent_2d_pca[y == c][:, 0],
                X_latent_2d_pca[y == c][:, 1],
                s=200,
                color=colors[c],
                marker=markers[c],
                label="class-" + str(c),
            )

        ax.set_title(title + " - PCA", fontsize=20)

        ax.legend(prop={"size": 20})
        fig.savefig(
            os.path.join(self.output_dir, pca_save_filename + ".pdf"),
            bbox_inches="tight",
        )

        plt.clf()

        fig = plt.figure(figsize=figsize)

        ax = fig.add_subplot(111)

        for c in range(n_classes):

            ax.scatter(
                X_latent_2d_tsne[y == c][:, 0],
                X_latent_2d_tsne[y == c][:, 1],
                s=200,
                color=colors[c],
                marker=markers[c],
                label="class-" + str(c),
            )

        ax.set_title(title + " - tSNE", fontsize=20)

        ax.legend(prop={"size": 20})
        fig.savefig(
            os.path.join(self.output_dir, tsne_save_filename + ".pdf"),
            bbox_inches="tight",
        )
