"""Van-Tran classifier."""

__all__ = ["APE", "VanTran_CLASSIFIER"]

import os
import time

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder as OHE

from deep_rehab_pile.classifiers.base import BASE_CLASSIFIER


class APE(tf.keras.layers.Layer):
    """Compute Absolute positional encoder.

    Parameters
    ----------
        d_model: int
            The embed dim (required).
        dropout: float, default = 0.1
            The dropout value.
        max_len: int, default = 1024
            The max. length of the incoming sequence.
    """

    def __init__(
        self,
        d_model: int,
        dropout_rate: float = 0.1,
        max_len: int = 1024,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        self.max_len = max_len
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        # Initialize positional encoding matrix with NumPy
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe = pe[np.newaxis, ...]

        # Convert to TensorFlow constant
        self.pe = tf.constant(pe, dtype=tf.float32)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x):
        """Add position information to input embedding.

        Parameters
        ----------
        x: KerasTensor
            The sequence fed to the positional encoder model.

        Returns
        -------
        KerasTensor
            The output with positional encoding and dropout.
        """
        x += self.pe[:, : tf.shape(x)[1], :]
        return self.dropout(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "dropout_rate": self.dropout_rate,
                "max_len": self.max_len,
            }
        )
        return config


class VanTran_CLASSIFIER(BASE_CLASSIFIER):
    """Depp Vanilla Transformer (VanTran) classifier.

    The encoder of the model proposed in [1]_ and
    the code is adapted from torch to tensorflow from:
    https://github.com/Mathux/ACTOR

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
    emb_size: int, default = 16
        Size of the embedding.
    dimm_ff: int, default = 256
        Dimensionality of the feed-forward layer.
    n_layers: int, default = 4
        The number of transformer encoder layers.
    num_heads: int, default = 8
        Number of attention heads.
    dropout_rate: float, default = 0.01
        Dropout rate for regularization.
    activation: str, default = "gelu"
        Activation function to use.
    epsilon: float, default = 1e-05
        Small value to avoid division by zero in normalization layers.
    batch_size: int, default = 64
        The batch size used for training.
    epochs: int, default = 3
        The number of epochs used to train the classifier.

    Returns
    -------
    None

    References
    ----------
    .. [1] Petrovich, M., Black, M. J., & Varol, G. (2021).
    Action-conditioned 3d human motion synthesis with transformer
    vae. In Proceedings of the IEEE/CVF International Conference
    on Computer Vision (pp. 10985-10995).
    """

    def __init__(
        self,
        output_dir: str,
        best_file_name: str,
        init_file_name: str,
        length_TS: int,
        n_joints: int,
        n_dim: int,
        emb_size: int = 256,
        dimm_ff: int = 1024,
        n_layers: int = 4,
        num_heads: int = 4,
        dropout_rate: float = 0.1,
        activation: str = "gelu",
        epsilon: float = 1e-6,
        batch_size: int = 64,
        epochs: int = 1500,
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

        self.emb_size = emb_size
        self.dimm_ff = dimm_ff
        self.n_layers = n_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.epsilon = epsilon

    def _build_model(self, return_model: bool = False, compile_model: bool = True):
        self.n_channels = self.n_joints * self.n_dim

        input_layer = tf.keras.layers.Input((self.length_TS, self.n_channels))

        embeddings = tf.keras.layers.Dense(units=self.emb_size, activation="linear")(
            input_layer
        )

        embeddings_with_positions = APE(
            d_model=self.emb_size,
            dropout_rate=self.dropout_rate,
            max_len=self.length_TS,
        )(embeddings)

        x = embeddings_with_positions

        for _ in range(self.n_layers):
            x = self._trasnformer_encoder_block(
                input_tensor=x,
                num_heads=self.num_heads,
                d_model=self.emb_size,
                dimm_ff=self.dimm_ff,
                dropout_rate=self.dropout_rate,
                activation=self.activation,
                epsilon=self.epsilon,
            )

        gap = tf.keras.layers.GlobalAveragePooling1D()(x)

        output_layer = tf.keras.layers.Dense(
            units=self.n_classes, activation="softmax"
        )(gap)

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

    def _trasnformer_encoder_block(
        self,
        input_tensor,
        num_heads,
        d_model,
        dimm_ff,
        dropout_rate,
        activation,
        epsilon,
    ):

        multihead_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model
        )(query=input_tensor, value=input_tensor, key=input_tensor)
        dropout1 = tf.keras.layers.Dropout(dropout_rate)(multihead_attention)
        layernorm1 = tf.keras.layers.LayerNormalization(epsilon=epsilon)(dropout1)
        res1 = tf.keras.layers.Add()([input_tensor, layernorm1])

        feed_forward1 = tf.keras.layers.Dense(units=dimm_ff, activation=activation)(
            res1
        )
        feed_forward2 = tf.keras.layers.Dense(units=d_model, activation="linear")(
            feed_forward1
        )
        dropout2 = tf.keras.layers.Dropout(dropout_rate)(feed_forward2)
        layernom2 = tf.keras.layers.LayerNormalization(epsilon=epsilon)(dropout2)
        res2 = tf.keras.layers.Add()([res1, layernom2])

        return res2

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

    def predict_proba(self, X: np.ndarray) -> tuple[np.ndarray, float]:
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
        """
        X = np.transpose(X, [0, 2, 1])
        custom_objects = {
            "APE": APE,
        }
        model = tf.keras.models.load_model(
            self.output_dir + self.best_file_name + ".keras",
            custom_objects=custom_objects,
            compile=False,
        )

        return model.predict(X)
