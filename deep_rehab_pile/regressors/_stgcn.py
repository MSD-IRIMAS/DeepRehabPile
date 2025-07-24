"""ST-GCN regressor."""

__all__ = ["GCNLayer", "MySqueezeLayer", "STGCN_REGRESSOR"]

import os
import time

import numpy as np
import tensorflow as tf

from deep_rehab_pile.regressors.base import BASE_REGRESSOR


class GCNLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        x, y = inputs
        return tf.einsum("ntvw,ntwc->ntvc", x, y)

    def get_config(self):
        config = super().get_config()
        return config


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


class STGCN_REGRESSOR(BASE_REGRESSOR):
    """Define STGCN regressor.

    This class implements a  Spatio-Temporal Graph
    Convolution Network based regressor proposed in [1]_
    and then adapted for human motion rehabilitation in [2]_.

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
    batch_size: int, optional, default=64
        Batch size for training.
    epochs: int, optional, default=1500
        Number of epochs for training.
    n_temporal_convolution_layers: int, default = 3
        The number of temporal convolution layers used in the
        GCN module.
    n_temporal_filters: int, default = 64
        The number of convolution filters used in each
        temporal convolution layer.
    temporal_kernel_size: list of int, default = [9, 15, 20]
        The size of kernel of each temporal convolution
        layer.
    n_bottleneck_filters: int, default = 64
        The number of convolution filters of the bottleneck
        layer.
    activation: str, default = "relu"
        The activation used in the convolution layers.
    dropout_rate: float, default = 0.25
        The dropout rate for regularization.
    n_lstm_layers: int, default = 4
        The number of LSTM layers used after the GCN module.
    n_lstm_units: list of int, default = [80, 40, 40, 80]
        The number of LSTM units per LSTM layer after
        the GCN module.

    References
    ----------
    .. [1] Yu, Bing, Haoteng Yin, and Zhanxing Zhu.
    "Spatio-temporal graph convolutional networks: A deep
    learning framework for traffic forecasting." arXiv
    preprint arXiv:1709.04875 (2017).
    .. [2]Deb, Swakshar, et al. "Graph convolutional networks
    for assessment of physical rehabilitation exercises." IEEE
    Transactions on Neural Systems and Rehabilitation
    Engineering 30 (2022): 410-419.
    """

    def __init__(
        self,
        output_dir: str,
        best_file_name: str,
        init_file_name: str,
        length_TS: int,
        n_joints: int,
        n_dim: int,
        kinematic_tree: list,
        batch_size: int = 64,
        epochs: int = 1500,
        n_temporal_convolution_layers: int = 3,
        n_temporal_filters: int = 64,
        temporal_kernel_size: list[int] = None,
        n_bottleneck_filters: int = 64,
        activation: str = "relu",
        dropout_rate: float = 0.25,
        n_lstm_layers: int = 4,
        n_lstm_units: list[int] = None,
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

        self.kinematic_tree = kinematic_tree
        self.n_temporal_convolution_layers = n_temporal_convolution_layers
        self.n_temporal_filters = n_temporal_filters
        self.temporal_kernel_size = temporal_kernel_size
        self.activation = activation
        self.n_bottleneck_filters = n_bottleneck_filters
        self.dropout_rate = dropout_rate
        self.n_lstm_layers = n_lstm_layers
        self.n_lstm_units = n_lstm_units

    def _build_model(self, return_model: bool = False, compile_model: bool = True):

        self.n_channels = self.n_joints * self.n_dim
        
        self.unroll = False
        if return_model:
            self.unroll = True # for flops count

        if self.temporal_kernel_size is None:
            if self.n_temporal_convolution_layers != 3:
                raise ValueError(
                    "Please provide the list of values of temporal_kernel_size"
                )
            else:
                self.temporal_kernel_size = [9, 15, 20]

        if self.n_lstm_units is None:
            if self.n_lstm_layers != 4:
                raise ValueError("Please provide the list of values of n_lstm_units")
            else:
                self.n_lstm_units = [80, 40, 40, 80]

        input_layer = tf.keras.layers.Input(shape=(self.length_TS, self.n_channels))
        reshape_layer = tf.keras.layers.Reshape(
            target_shape=(self.length_TS, self.n_joints, self.n_dim)
        )(input_layer)

        sgcn1 = self._sgcn_module(
            input_tensor=reshape_layer,
            dropout_rate=self.dropout_rate,
            n_temporal_convolution_layers=self.n_temporal_convolution_layers,
            n_temporal_filters=self.n_temporal_filters,
            temporal_kernel_size=self.temporal_kernel_size,
            activation=self.activation,
            n_bottleneck_filters=self.n_bottleneck_filters,
        )
        sgcn2 = self._sgcn_module(
            input_tensor=sgcn1,
            dropout_rate=self.dropout_rate,
            n_temporal_convolution_layers=self.n_temporal_convolution_layers,
            n_temporal_filters=self.n_temporal_filters,
            temporal_kernel_size=self.temporal_kernel_size,
            activation=self.activation,
            n_bottleneck_filters=self.n_bottleneck_filters,
        )

        sgcn_residual1 = tf.keras.layers.Add()([sgcn1, sgcn2])

        sgcn3 = self._sgcn_module(
            input_tensor=sgcn_residual1,
            dropout_rate=self.dropout_rate,
            n_temporal_convolution_layers=self.n_temporal_convolution_layers,
            n_temporal_filters=self.n_temporal_filters,
            temporal_kernel_size=self.temporal_kernel_size,
            activation=self.activation,
            n_bottleneck_filters=self.n_bottleneck_filters,
        )

        sgcn_residual2 = tf.keras.layers.Add()([sgcn_residual1, sgcn3])

        x = sgcn_residual2

        for i in range(self.n_lstm_layers):
            _return_sequences = True if i < self.n_lstm_layers - 1 else False
            x = self._lstm_module(
                input_tensor=x,
                units=self.n_lstm_units[i],
                return_sequences=_return_sequences,
                dropout_rate=self.dropout_rate,
            )

        output_layer = tf.keras.layers.Dense(units=1, activation="linear")(x)

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

    def _temporal_convolution(
        self, input_tensor, use_residual, filters, kernel_size, activation, padding
    ):
        temporal_conv = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding=padding,
            activation=activation,
        )(input_tensor)

        if use_residual:
            concat_input_temporal_conv = tf.keras.layers.concatenate(
                [input_tensor, temporal_conv], axis=-1
            )
            return concat_input_temporal_conv
        else:
            return temporal_conv

    def _graph_conv_hop_localization_module(
        self, input_tensor, biases, bottleneck_filters, activation
    ):

        bottleneck = tf.keras.layers.Conv2D(
            filters=bottleneck_filters,
            kernel_size=(1, 1),
            activation=activation,
        )(input_tensor)

        # transform it to a video shape data
        expand_bottleneck = tf.keras.layers.Reshape(
            target_shape=(self.length_TS, self.n_joints, 1, bottleneck_filters)
        )(bottleneck)

        conv2d_lstm = tf.keras.layers.ConvLSTM2D(
            filters=self.n_joints,
            kernel_size=(1, 1),
            return_sequences=True,
            unroll=self.unroll
        )(expand_bottleneck)
        conv2d_lstm = MySqueezeLayer(axis=3)(conv2d_lstm)

        conv2d_lstm_leaky_relu = tf.keras.layers.Activation(activation="leaky_relu")(
            conv2d_lstm
        )
        conv2d_lstm_residual_with_biases = biases + conv2d_lstm_leaky_relu
        coefs = tf.keras.layers.Activation(activation="softmax")(
            conv2d_lstm_residual_with_biases
        )

        graph_conv_on_bottleneck = GCNLayer()([coefs, bottleneck])

        return graph_conv_on_bottleneck

    def _sgcn_module(
        self,
        input_tensor,
        dropout_rate,
        n_temporal_convolution_layers,
        n_temporal_filters,
        temporal_kernel_size,
        activation,
        n_bottleneck_filters,
    ):

        temporal_convolution_with_residual = self._temporal_convolution(
            input_tensor=input_tensor,
            use_residual=True,
            filters=n_temporal_filters,
            kernel_size=(temporal_kernel_size[0], 1),
            padding="same",
            activation=activation,
        )
        graph_hop_first = self._graph_conv_hop_localization_module(
            input_tensor=temporal_convolution_with_residual,
            biases=self.bias_mat_1,
            bottleneck_filters=n_bottleneck_filters,
            activation=activation,
        )
        graph_hop_second = self._graph_conv_hop_localization_module(
            input_tensor=temporal_convolution_with_residual,
            biases=self.bias_mat_2,
            bottleneck_filters=n_bottleneck_filters,
            activation=activation,
        )

        graph_convolution = tf.keras.layers.concatenate(
            [graph_hop_first, graph_hop_second], axis=-1
        )

        x = graph_convolution
        temporal_convolution_list = []

        for i in range(n_temporal_convolution_layers):
            x = self._temporal_convolution(
                input_tensor=x,
                use_residual=False,
                filters=n_temporal_filters // 4,
                kernel_size=(temporal_kernel_size[i], 1),
                padding="same",
                activation=activation,
            )

            x = tf.keras.layers.Dropout(dropout_rate)(x)

            temporal_convolution_list.append(x)

        if len(temporal_convolution_list) > 1:
            concatenate_convolutions = tf.keras.layers.concatenate(
                temporal_convolution_list, axis=-1
            )
            return concatenate_convolutions
        else:
            return temporal_convolution_list[0]

    def _lstm_module(self, input_tensor, units, return_sequences, dropout_rate):

        if len(input_tensor.shape) == 4:
            reshape_layer = tf.keras.layers.Reshape(
                target_shape=(
                    self.length_TS,
                    input_tensor.shape[-2] * input_tensor.shape[-1],
                )
            )(input_tensor)
        else:
            reshape_layer = input_tensor

        lstm = tf.keras.layers.LSTM(units=units, return_sequences=return_sequences,
                                    unroll=self.unroll)(
            reshape_layer
        )
        dropout = tf.keras.layers.Dropout(dropout_rate)(lstm)

        return dropout

    def _build_graph(self):

        self_link = [(i, i) for i in range(self.n_joints)]
        neighbor_link = [
            (self.kinematic_tree[i][j], self.kinematic_tree[i][j + 1])
            for i in range(len(self.kinematic_tree))
            for j in range(len(self.kinematic_tree[i]) - 1)
        ]
        edge = self_link + neighbor_link

        self._adjacency_matrix = np.zeros((self.n_joints, self.n_joints))
        self._second_order_adjacency_matrix = np.zeros((self.n_joints, self.n_joints))

        for i, j in edge:
            self._adjacency_matrix[j, i] = 1
            self._adjacency_matrix[i, j] = 1

        for root in range(self.n_joints):
            for neighbour in range(self.n_joints):
                if self._adjacency_matrix[root, neighbour] == 1:
                    for neighbour_of_neigbour in range(self.n_joints):
                        if (
                            self._adjacency_matrix[neighbour, neighbour_of_neigbour]
                            == 1
                        ):
                            self._second_order_adjacency_matrix[
                                root, neighbour_of_neigbour
                            ] = 1

        self.bias_mat_1 = np.zeros(self._adjacency_matrix.shape)
        self.bias_mat_2 = np.zeros(self._second_order_adjacency_matrix.shape)

        self.bias_mat_1 = np.where(self._adjacency_matrix != 0, self.bias_mat_1, -1e9)
        self.bias_mat_2 = np.where(
            self._second_order_adjacency_matrix != 0,
            self._second_order_adjacency_matrix,
            -1e9,
        )

        self.adjacency_matrix = self._adjacency_matrix.astype("float32")
        self.second_order_adjacency_matrix = self._second_order_adjacency_matrix.astype(
            "float32"
        )
        self.adjacency_matrix = tf.convert_to_tensor(self.adjacency_matrix)
        self.second_order_adjacency_matrix = tf.convert_to_tensor(
            self.second_order_adjacency_matrix
        )

        self.bias_mat_1 = self.bias_mat_1.astype("float32")
        self.bias_mat_2 = self.bias_mat_2.astype("float32")
        self.bias_mat_1 = tf.convert_to_tensor(self.bias_mat_1)
        self.bias_mat_2 = tf.convert_to_tensor(self.bias_mat_2)

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

        self._build_graph()
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
            The predicted score labels.
        delta_time: float,
            The inference time.
        """
        start_time = time.time()
        X = np.transpose(X, [0, 2, 1])
        custom_objects = {
            "GCNLayer": GCNLayer,
            "MySqueezeLayer": MySqueezeLayer,
        }
        model = tf.keras.models.load_model(
            self.output_dir + self.best_file_name + ".keras",
            custom_objects=custom_objects,
            compile=False,
        )

        preds = model.predict(X).reshape((len(X),))

        preds[preds > 1.0] = 1.0
        preds[preds < 0.0] = 0.0

        delta_time = time.time() - start_time

        return preds, delta_time
