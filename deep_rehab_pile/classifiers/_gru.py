"""GRU classifier."""

__all__ = ["GRU_CLASSIFIER"]

import os
import time

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder as OHE

from deep_rehab_pile.classifiers.base import BASE_CLASSIFIER


class GRU_CLASSIFIER(BASE_CLASSIFIER):
    """Depp GRU classifier.

    Code adapted from original work of [1]_ and taken from [2]_.

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
    activation: str, default = 'tanh'
        The activation function used in the GRU cell.
    hidden_units_gru: int, default = 128
        The size of the GRU cell.
    hidden_units_fc: int, default = 30
        The size of the fc latent space.
    batch_size: int, default = 64,
        The batch size used for training.
    epochs: int, default = 3,
        The number of epochs used to train the classifier.

    Returns
    -------
    None

    References
    ----------
    .. [1] Guo, Chuan, et al. "Action2motion: Conditioned generation of 3d human
    motions." Proceedings of the 28th ACM International Conference on
    Multimedia. 2020.
    .. [2] Ismail-Fawaz et al. " A Supervised Variational Auto-Encoder for
    Human Motion Generation using Convolutional Neural Networks."
    International Conference on Pattern Recognition and Artificial
    Intelligence (ICPRAI). 2024.
    """

    def __init__(
        self,
        output_dir: str,
        best_file_name: str,
        init_file_name: str,
        length_TS: int,
        n_joints: int,
        n_dim: int,
        activation: str = "tanh",
        hidden_units_gru: int = 128,
        hidden_units_fc: int = 30,
        batch_size: int = 64,
        epochs: int = 3,
    ):
        super().__init__(
            output_dir=output_dir,
            best_file_name=best_file_name,
            init_file_name=init_file_name,
            length_TS=length_TS,
            n_joints=n_joints,
            n_dim=n_dim,
            batch_size=batch_size,
            epochs=epochs,
        )

        self.activation = activation
        self.hidden_units_gru = hidden_units_gru
        self.hidden_units_fc = hidden_units_fc

    def _build_model(self, return_model: bool = False, compile_model: bool = True):
        self.n_channels = self.n_joints * self.n_dim

        input_layer = tf.keras.layers.Input((self.length_TS, self.n_channels))
        
        unroll = False
        if return_model:
            unroll = True # for flops count

        gru_1 = tf.keras.layers.GRU(
            units=self.hidden_units_gru,
            activation=self.activation,
            return_sequences=True,
            unroll=unroll
        )(input_layer)

        gru_2 = tf.keras.layers.GRU(
            units=self.hidden_units_gru,
            activation=self.activation,
            return_sequences=False,
            unroll=unroll
        )(gru_1)

        hidden_layer = tf.keras.layers.Dense(
            units=self.hidden_units_fc, activation=self.activation
        )(gru_2)

        output_layer = tf.keras.layers.Dense(
            units=self.n_classes, activation="softmax"
        )(hidden_layer)

        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

        if compile_model:
            model.compile(loss="categorical_crossentropy", optimizer="Adam")

            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor="loss", factor=0.5, patience=50, min_lr=1e-4
            )

            model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath=self.output_dir + self.best_file_name + ".keras",
                monitor="loss",
                save_best_only=True,
            )

            self.callbacks = [reduce_lr, model_checkpoint]

        if return_model:
            return model
        else:
            self.model = model

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
        X = np.transpose(X, [0, 2, 1])
        self.n_classes = len(np.unique(y))

        self._build_model()

        self.model.save(os.path.join(self.output_dir, self.init_file_name + ".keras"))

        ohe = OHE(sparse_output=False)
        y_ohe = np.expand_dims(y, axis=-1)
        y_ohe = ohe.fit_transform(y_ohe)

        self.model.fit(
            X,
            y_ohe,
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=self.callbacks,
            verbose=1,
        )
        delta_time = time.time() - start_time

        tf.keras.backend.clear_session()

        return delta_time
