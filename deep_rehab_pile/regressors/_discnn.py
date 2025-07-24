"""DisjointCNN regressor."""

__all__ = ["DisjointCNN_REGRESSOR"]

import os
import time

import numpy as np
import tensorflow as tf

from deep_rehab_pile.regressors.base import BASE_REGRESSOR


class DisjointCNN_REGRESSOR(BASE_REGRESSOR):
    """Depp DisjointCNN regressor.

    The model is proposed in [1]_ to apply convolutions
    specifically for multivariate series, temporal-spatial
    phases. It has been adopted for regression here.

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
    n_filters: omt. default = 64
        The number of convolution filters.
    kernel_size: list, default = None
        The kernel size of convolution layers. If None
        then list is default to [8,5,5,3].
    pool_size: int, default = 5
        The size of the max pool layer.
    hidden_fc_units: int, default = 128
        The number of fully connected units.
    activation_fc: str, default = "relu"
        The activation of the fully connected layer.
    batch_size: int, default = 64,
        The batch size used for training.
    epochs: int, default = 3,
        The number of epochs used to train the regressor.

    Returns
    -------
    None

    References
    ----------
    .. [1] Foumani, Seyed Navid Mohammadi, Chang Wei Tan, and Mahsa Salehi.
    "Disjoint-cnn for multivariate time series classification."
    2021 International Conference on Data Mining Workshops
    (ICDMW). IEEE, 2021.
    """

    def __init__(
        self,
        output_dir: str,
        best_file_name: str,
        init_file_name: str,
        length_TS: int,
        n_joints: int,
        n_dim: int,
        n_filters: int = 64,
        kernel_size: list = None,
        pool_size: int = 5,
        hidden_fc_units: int = 128,
        activation_fc: str = "relu",
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

        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.hidden_fc_units = hidden_fc_units
        self.activation_fc = activation_fc

    def _build_model(self, return_model: bool = False, compile_model: bool = True):
        self.n_channels = self.n_joints * self.n_dim

        input_layer = tf.keras.layers.Input((self.length_TS, self.n_channels))
        reshape_layer = tf.keras.layers.Reshape(
            target_shape=(self.length_TS, self.n_channels, 1)
        )(input_layer)

        conv1 = tf.keras.layers.Conv2D(
            self.n_filters,
            (self._kernel_size[0], 1),
            padding="same",
            kernel_initializer="he_uniform",
        )(reshape_layer)
        conv1 = tf.keras.layers.BatchNormalization()(conv1)
        conv1 = tf.keras.layers.ELU(alpha=1.0)(conv1)
        conv1 = tf.keras.layers.Conv2D(
            self.n_filters,
            (1, self.n_channels),
            padding="valid",
            kernel_initializer="he_uniform",
        )(conv1)
        conv1 = tf.keras.layers.BatchNormalization()(conv1)
        conv1 = tf.keras.layers.ELU(alpha=1.0)(conv1)
        conv1 = tf.keras.layers.Permute((1, 3, 2))(conv1)

        conv2 = tf.keras.layers.Conv2D(
            self.n_filters,
            (self._kernel_size[1], 1),
            padding="same",
            kernel_initializer="he_uniform",
        )(conv1)
        conv2 = tf.keras.layers.BatchNormalization()(conv2)
        conv2 = tf.keras.layers.ELU(alpha=1.0)(conv2)
        conv2 = tf.keras.layers.Conv2D(
            self.n_filters,
            (1, conv2.shape[2]),
            padding="valid",
            kernel_initializer="he_uniform",
        )(conv2)
        conv2 = tf.keras.layers.BatchNormalization()(conv2)
        conv2 = tf.keras.layers.ELU(alpha=1.0)(conv2)
        conv2 = tf.keras.layers.Permute((1, 3, 2))(conv2)

        conv3 = tf.keras.layers.Conv2D(
            self.n_filters,
            (self._kernel_size[2], 1),
            padding="same",
            kernel_initializer="he_uniform",
        )(conv2)
        conv3 = tf.keras.layers.BatchNormalization()(conv3)
        conv3 = tf.keras.layers.ELU(alpha=1.0)(conv3)
        conv3 = tf.keras.layers.Conv2D(
            self.n_filters,
            (1, conv3.shape[2]),
            padding="valid",
            kernel_initializer="he_uniform",
        )(conv2)
        conv3 = tf.keras.layers.BatchNormalization()(conv3)
        conv3 = tf.keras.layers.ELU(alpha=1.0)(conv3)
        conv3 = tf.keras.layers.Permute((1, 3, 2))(conv3)

        conv4 = tf.keras.layers.Conv2D(
            self.n_filters,
            (self._kernel_size[3], 1),
            padding="same",
            kernel_initializer="he_uniform",
        )(conv3)
        conv4 = tf.keras.layers.BatchNormalization()(conv4)
        conv4 = tf.keras.layers.ELU(alpha=1.0)(conv4)
        conv4 = tf.keras.layers.Conv2D(
            self.n_filters,
            (1, conv4.shape[2]),
            padding="valid",
            kernel_initializer="he_uniform",
        )(conv4)
        conv4 = tf.keras.layers.BatchNormalization()(conv4)
        conv4 = tf.keras.layers.ELU(alpha=1.0)(conv4)

        max_pool = tf.keras.layers.MaxPooling2D(
            pool_size=(self.pool_size, 1), strides=None, padding="valid"
        )(conv4)
        gap = tf.keras.layers.GlobalAveragePooling2D()(max_pool)

        fc = tf.keras.layers.Dense(self.hidden_fc_units, activation=self.activation_fc)(
            gap
        )

        output_layer = tf.keras.layers.Dense(units=1, activation="linear")(fc)

        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

        if compile_model:
            model.compile(loss="mse", optimizer="Adam")

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
        Train the regressor.

        Parameters
        ----------
        X: np.ndarray, shape = (n_samples, n_channels, n_timepoints),
            The input samples.
        y: np.ndarray, shape = (n_samples),
            The score labels.

        Returns
        -------
        delta_time: float,
            The training time.
        """
        start_time = time.time()
        X = np.transpose(X, [0, 2, 1])

        if self.kernel_size is None:
            self._kernel_size = [8, 5, 5, 3]
        else:
            if isinstance(self.kernel_size, int):
                self._kernel_size = [self.kernel_size] * 4
            elif isinstance(self.kernel_size, list):
                if len(self.kernel_size) == 4 or len(self.kernel_size) == 1:
                    self._kernel_size = self.kernel_size
                else:
                    raise ValueError(
                        "The length of the kernel_size list should be 4,",
                        "or 1 to broadcast or one value int.",
                    )

        self._build_model()

        self.model.save(os.path.join(self.output_dir, self.init_file_name + ".keras"))

        self.model.fit(
            X,
            y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=self.callbacks,
            verbose=1,
        )
        delta_time = time.time() - start_time

        tf.keras.backend.clear_session()

        return delta_time
