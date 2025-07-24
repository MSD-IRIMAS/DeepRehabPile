"""Conv-LSTM classifier."""

__all__ = ["ConvLSTM_CLASSIFIER"]

import os
import time

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder as OHE

from deep_rehab_pile.classifiers.base import BASE_CLASSIFIER


class ConvLSTM_CLASSIFIER(BASE_CLASSIFIER):
    """Depp Conv-LSTM classifier.

    The model is proposed in [1]_ and the code is adapted from:
    https://github.com/takumiw/Deep-Learning-for-Human-Activity-Recognition/.

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
    activation_conv: str, default = 'relu'
        The activation function used in the convolution layer.
    activation_lstm: str, default = 'tanh'
        The activation function used in the LSTM cell.
    n_conv_filters: omt. default = 64
        The number of convolution filters.
    kernel_size: int, default = 5
        The kernel size of convolution layers.
    hidden_units_lstm: int, default = 128
        The size of the LSTM cell.
    dropout_rate: float, default = 0.5
        The rate of the dropout layer.
    batch_size: int, default = 64,
        The batch size used for training.
    epochs: int, default = 3,
        The number of epochs used to train the classifier.

    Returns
    -------
    None

    References
    ----------
    .. [1] Ordóñez, Francisco Javier, and Daniel Roggen.
    "Deep convolutional and lstm recurrent neural networks for
    multimodal wearable activity recognition." Sensors 16.1 (2016): 115.
    """

    def __init__(
        self,
        output_dir: str,
        best_file_name: str,
        init_file_name: str,
        length_TS: int,
        n_joints: int,
        n_dim: int,
        activation_conv: str = "relu",
        activation_lstm: str = "tanh",
        n_conv_filters: int = 64,
        kernel_size: int = 5,
        hidden_units_lstm: int = 128,
        dropout_rate: int = 0.5,
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

        self.activation_conv = activation_conv
        self.activation_lstm = activation_lstm
        self.n_conv_filters = n_conv_filters
        self.kernel_size = kernel_size
        self.hidden_units_lstm = hidden_units_lstm
        self.dropout_rate = dropout_rate

    def _build_model(self, return_model: bool = False, compile_model: bool = True):
        self.n_channels = self.n_joints * self.n_dim

        input_layer = tf.keras.layers.Input((self.length_TS, self.n_channels))
        reshape_layer = tf.keras.layers.Reshape(
            target_shape=(self.length_TS, self.n_channels, 1)
        )(input_layer)

        conv1 = tf.keras.layers.Conv2D(
            filters=self.n_conv_filters,
            kernel_size=(self.kernel_size, 1),
        )(reshape_layer)
        conv1 = tf.keras.layers.Activation(activation=self.activation_conv)(conv1)

        conv2 = tf.keras.layers.Conv2D(
            filters=self.n_conv_filters,
            kernel_size=(self.kernel_size, 1),
        )(conv1)
        conv2 = tf.keras.layers.Activation(activation=self.activation_conv)(conv2)

        conv3 = tf.keras.layers.Conv2D(
            filters=self.n_conv_filters,
            kernel_size=(self.kernel_size, 1),
        )(conv2)
        conv3 = tf.keras.layers.Activation(activation=self.activation_conv)(conv3)

        conv4 = tf.keras.layers.Conv2D(
            filters=self.n_conv_filters,
            kernel_size=(self.kernel_size, 1),
        )(conv3)
        conv4 = tf.keras.layers.Activation(activation=self.activation_conv)(conv4)

        conv_reshaped = tf.keras.layers.Reshape(
            target_shape=(-1, self.n_channels * self.n_conv_filters)
        )(conv4)
        
        unroll = False
        if return_model:
            unroll = True # for flops count

        lstm1 = tf.keras.layers.LSTM(
            units=self.hidden_units_lstm,
            activation=self.activation_lstm,
            return_sequences=True,
            unroll=unroll
        )(conv_reshaped)
        lstm1 = tf.keras.layers.Dropout(self.dropout_rate)(lstm1)

        lstm2 = tf.keras.layers.LSTM(
            units=self.hidden_units_lstm,
            activation=self.activation_lstm,
            return_sequences=False,
            unroll=unroll
        )(lstm1)
        lstm2 = tf.keras.layers.Dropout(self.dropout_rate)(lstm2)

        output_layer = tf.keras.layers.Dense(
            units=self.n_classes, activation="softmax"
        )(lstm2)

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

        ohe = OHE(sparse_output=False)
        y_ohe = np.expand_dims(y, axis=-1)
        y_ohe = ohe.fit_transform(y_ohe)

        self.model.save(os.path.join(self.output_dir, self.init_file_name + ".keras"))

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
