"""ConvTran classifier."""

__all__ = ["tAPE", "Attention_Rel_Scl", "MySqueezeLayer", "ConvTran_CLASSIFIER"]

import os
import time

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder as OHE

from deep_rehab_pile.classifiers.base import BASE_CLASSIFIER


class tAPE(tf.keras.layers.Layer):
    """Compute time Absolute positional encoder.

    Inject some information about the relative or absolute position of the tokens
    in the sequence. The positional encodings have the same dimension as
    the embeddings, so that the two can be summed. Here, we use sine and cosine
    functions of different frequencies. This is taken from [1]_.

    Parameters
    ----------
        d_model: int
            The embed dim (required).
        dropout: float, default = 0.1
            The dropout value.
        max_len: int, default = 1024
            The max. length of the incoming sequence.
        scale_factor: float, default = 1.0
            The scaling factor for the positional encoding.
    """

    def __init__(
        self,
        d_model: int,
        dropout_rate: float = 0.1,
        max_len: int = 1024,
        scale_factor: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        self.max_len = max_len
        self.scale_factor = scale_factor
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        # Initialize positional encoding matrix with NumPy
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin((position * div_term) * (d_model / max_len))
        pe[:, 1::2] = np.cos((position * div_term) * (d_model / max_len))
        pe = scale_factor * pe[np.newaxis, ...]

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
                "scale_factor": self.scale_factor,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Attention_Rel_Scl(tf.keras.layers.Layer):
    """
    A TensorFlow Keras layer implementing relative scaling multi-head attention.

    This layer performs multi-head attention with relative position encoding, a method
    that incorporates information about the positions of elements within a sequence. It
    includes mechanisms for creating and applying relative biases based on sequence
    positions.

    Parameters
    ----------
    emb_size: int
        The size of the embeddings.
    num_heads: int
        The number of attention heads.
    seq_len: int
        The length of the input sequences.
    dropout_rate: float
        The dropout rate to apply during training.
    **kwargs: dict
        Additional keyword arguments to be passed to the `Layer` superclass.
    """

    def __init__(
        self, emb_size: int, num_heads: int, seq_len: int, dropout_rate: float, **kwargs
    ):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.emb_size = emb_size
        self.dropout_rate = dropout_rate
        self.scale = emb_size**-0.5

        self.key = tf.keras.layers.Dense(emb_size, use_bias=False)
        self.value = tf.keras.layers.Dense(emb_size, use_bias=False)
        self.query = tf.keras.layers.Dense(emb_size, use_bias=False)

        self.relative_bias_table = self.add_weight(
            shape=(2 * seq_len - 1, num_heads), initializer="zeros", trainable=True
        )

        range_tensor = tf.range(seq_len)
        relative_coords = tf.expand_dims(range_tensor, axis=0) - tf.expand_dims(
            range_tensor, axis=1
        )
        relative_index = tf.clip_by_value(
            relative_coords + (seq_len - 1), 0, 2 * seq_len - 2
        )
        self.relative_index = relative_index

        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.to_out = tf.keras.layers.LayerNormalization()

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x):
        """Compute attention layer with relative positional encoding.

        Parameters
        ----------
        x: KerasTensor
            The sequence fed to the attention layer.

        Returns
        -------
        KerasTensor
            The output of the attention layer.
        """
        batch_size, seq_len, _ = x.shape

        k = tf.keras.layers.Reshape(target_shape=(seq_len, self.num_heads, -1))(
            self.key(x)
        )
        k = tf.transpose(k, perm=[0, 2, 3, 1])

        v = tf.keras.layers.Reshape(target_shape=(seq_len, self.num_heads, -1))(
            self.value(x)
        )
        v = tf.transpose(v, perm=[0, 2, 1, 3])

        q = tf.keras.layers.Reshape(target_shape=(seq_len, self.num_heads, -1))(
            self.query(x)
        )
        q = tf.transpose(q, perm=[0, 2, 1, 3])

        attn = tf.matmul(q, k) * self.scale
        attn = tf.nn.softmax(attn, axis=-1)

        relative_bias = tf.gather(self.relative_bias_table, self.relative_index)
        relative_bias = tf.reshape(relative_bias, (-1, self.num_heads, 1))
        relative_bias = tf.transpose(relative_bias, perm=[2, 0, 1])
        relative_bias = tf.reshape(
            relative_bias, (self.num_heads, 1 * self.seq_len, 1 * self.seq_len)
        )

        attn += relative_bias

        out = tf.matmul(attn, v)
        out = tf.transpose(out, perm=[0, 2, 1, 3])

        out = tf.keras.layers.Reshape(target_shape=(self.seq_len, -1))(out)

        out = self.to_out(out)

        return out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "emb_size": self.emb_size,
                "num_heads": self.num_heads,
                "seq_len": self.seq_len,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class MySqueezeLayer(tf.keras.layers.Layer):
    """Define the squeeze layer.

    Parameters
    ----------
    axis: int
        The axis to squeeze.
    """

    def __init__(self, axis: int, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x):
        """Compute the squeeze operation.

        Parameters
        ----------
        x: KerasTensor
            The input data.

        Returns
        -------
        KerasTensor
            The squeeze version of x on axis.
        """
        return tf.squeeze(x, axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class ConvTran_CLASSIFIER(BASE_CLASSIFIER):
    """Define Convolutional Transformer Classifier.

    This class implements a convolutional transformer-based classifier
    proposed in [1]_.

    Parameters
    ----------
    output_dir: str
        Directory to save the output results.
    best_file_name: str
        Name of the file to save the best model.
    init_file_name: str,
        The name of the init model to save.
    length_TS: int,
        The length of input skeleton sequence.
    n_joints: int,
        The number of joints in the skeleton.
    n_dim: int,
        The number of dimensions per joint.
    kernel_size: int = 8, optional
        Size of the convolutional kernel.
    factor_filters: int = 4, optional
        Factor to determine the number of filters.
    activation: str = "gelu", optional
        Activation function to use..
    dropout_rate: float = 0.01, optional
        Dropout rate for regularization.
    emb_size: int = 16, optional
        Size of the embedding.
    num_heads: int = 8, optional
        Number of attention heads.
    epsilon: float = 1e-05, optional
        Small value to avoid division by zero in normalization layers.
    dimm_ff: int = 256, optional
        Dimensionality of the feed-forward layer.
    activation_ff: str = "relu", optional
        Activation function for the feed-forward layer.
    batch_size: int = 64, optional
        Batch size for training.
    epochs: int = 1500, optional
        Number of epochs for training.

    References
    ----------
    .. [1] Foumani, Navid Mohammadi, et al. "Improving position encoding of
    transformers for multivariate time series classification." Data Mining
    and Knowledge Discovery 38.1 (2024): 22-48.
    """

    def __init__(
        self,
        output_dir: str,
        best_file_name: str,
        init_file_name: str,
        length_TS: int,
        n_joints: int,
        n_dim: int,
        kernel_size: int = 8,
        factor_filters: int = 4,
        activation: str = "gelu",
        dropout_rate: float = 0.01,
        emb_size: int = 16,
        num_heads: int = 8,
        epsilon: float = 1e-5,
        dimm_ff: int = 256,
        activation_ff: str = "relu",
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

        self.kernel_size = kernel_size
        self.factor_filters = factor_filters
        self.activation = activation

        self.dropout_rate = dropout_rate
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.epsilon = epsilon

        self.dimm_ff = dimm_ff
        self.activation_ff = activation_ff

    def _build_model(self, return_model: bool = False, compile_model: bool = True):
        with tf.device("/GPU:0"):
            self.n_channels = self.n_joints * self.n_dim

            input_layer = tf.keras.layers.Input((self.length_TS, self.n_channels))

            adding_img_channel_axis = tf.keras.layers.Reshape(
                target_shape=(self.length_TS, self.n_channels, 1)
            )(input_layer)

            # Disjoint CNN

            temporal_conv = tf.keras.layers.Conv2D(
                filters=self.emb_size * self.factor_filters,
                kernel_size=(self.kernel_size, 1),
                padding="same",
            )(adding_img_channel_axis)
            batch_norm1 = tf.keras.layers.BatchNormalization()(temporal_conv)
            activation1 = tf.keras.layers.Activation(activation=self.activation)(
                batch_norm1
            )

            spatial_conv = tf.keras.layers.Conv2D(
                filters=self.emb_size, kernel_size=(1, self.n_channels), padding="valid"
            )(activation1)
            batch_norm2 = tf.keras.layers.BatchNormalization()(spatial_conv)
            activation2 = tf.keras.layers.Activation(activation=self.activation)(
                batch_norm2
            )

            # embeddings = tf.squeeze(activation2, axis=2)
            embeddings = MySqueezeLayer(axis=2)(activation2)

            embeddings_with_positions = tAPE(
                d_model=self.emb_size,
                dropout_rate=self.dropout_rate,
                max_len=self.length_TS,
            )(embeddings)

            attention_layer = Attention_Rel_Scl(
                emb_size=self.emb_size,
                num_heads=self.num_heads,
                seq_len=self.length_TS,
                dropout_rate=self.dropout_rate,
            )(embeddings_with_positions)

            residual_layer1 = tf.keras.layers.Add()(
                [embeddings_with_positions, attention_layer]
            )

            normalization_layer1 = tf.keras.layers.LayerNormalization(
                epsilon=self.epsilon
            )(residual_layer1)

            ff_layer1 = tf.keras.layers.Dense(units=self.dimm_ff)(normalization_layer1)
            ff_layer2 = tf.keras.layers.Activation(activation=self.activation_ff)(
                ff_layer1
            )
            ff_layer3 = tf.keras.layers.Dropout(self.dropout_rate)(ff_layer2)
            ff_layer4 = tf.keras.layers.Dense(units=self.emb_size)(ff_layer3)
            ff_layer5 = tf.keras.layers.Dropout(self.dropout_rate)(ff_layer4)

            residual_layer2 = tf.keras.layers.Add()([normalization_layer1, ff_layer5])

            normalization_layer2 = tf.keras.layers.LayerNormalization(
                epsilon=self.epsilon
            )(residual_layer2)

            gap_layer = tf.keras.layers.GlobalAveragePooling1D()(normalization_layer2)

            output_layer = tf.keras.layers.Dense(
                units=self.n_classes, activation="softmax"
            )(gap_layer)

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

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> float:
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
            "tAPE": tAPE,
            "Attention_Rel_Scl": Attention_Rel_Scl,
            "MySqueezeLayer": MySqueezeLayer,
        }
        model = tf.keras.models.load_model(
            self.output_dir + self.best_file_name + ".keras",
            custom_objects=custom_objects,
            compile=False,
        )

        return model.predict(X)
