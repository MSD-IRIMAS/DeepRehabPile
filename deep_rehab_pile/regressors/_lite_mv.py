"""LITE Multivariate regressor."""

__all__ = ["LITE_MV_REGRESSOR"]

import os
import time

import numpy as np
import tensorflow as tf

from deep_rehab_pile.regressors.base import BASE_REGRESSOR


class LITE_MV_REGRESSOR(BASE_REGRESSOR):
    """Define LITE Multivariate Regressor.

    This class implements a LITE Multivariate network-based regressor
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
    batch_size: int, optional, default=64
        Batch size for training.
    epochs: int, optional, default=1500
        Number of epochs for training.
    n_filters : int or list of int32, default = 32
        The number of filters used in one lite layer, if not a list, the same
        number of filters is used in all lite layers.
    kernel_size : int or list of int, default = 40
        The head kernel size used for each lite layer, if not a list, the same
        is used in all lite layers.
    strides : int or list of int, default = 1
        The strides of kernels in convolution layers for each lite layer,
        if not a list, the same is used in all lite layers.
    activation : str or list of str, default = 'relu'
        The activation function used in each lite layer, if not a list,
        the same is used in all lite layers.

    References
    ----------
    .. [1] Ismail-Fawaz, Ali, et al. "Look Into the LITE
    in Deep Learning for Time Series Classification."
    arXiv preprint arXiv:2409.02869 (2024).
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
        n_filters: int = 32,
        kernel_size: int = 40,
        strides: int = 1,
        activation: str = "relu",
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
        self.strides = strides
        self.activation = activation

    def _build_model(self, return_model: bool = False, compile_model: bool = True):
        self.n_channels = self.n_joints * self.n_dim

        input_layer = tf.keras.layers.Input((self.length_TS, self.n_channels))

        inception = self._inception_module(
            input_tensor=input_layer,
            dilation_rate=1,
            use_custom_filters=True,
            use_multiplexing=True,
        )

        _kernel_size = self.kernel_size // 2

        input_tensor = inception

        dilation_rate = 1

        for i in range(2):
            dilation_rate = 2 ** (i + 1)

            x = self._fcn_module(
                input_tensor=input_tensor,
                kernel_size=_kernel_size // (2**i),
                n_filters=self.n_filters,
                dilation_rate=dilation_rate,
            )

            input_tensor = x

        gap = tf.keras.layers.GlobalAveragePooling1D()(x)

        output_layer = tf.keras.layers.Dense(units=1, activation="linear")(gap)

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

    def hybrid_layer(self, input_tensor, input_channels, kernel_sizes=None):
        """Construct the hybrid layer to compute features of custom filters.

        Parameters
        ----------
        input_tensor : tensorflow tensor, usually the input layer of the model.
        input_channels : int, the number of input channels in case of multivariate.
        kernel_sizes : list of int, default = [2,4,8,16,32,64],
        the size of the hand-crafted filters.

        Returns
        -------
        hybrid_layer : tensorflow tensor containing the concatenation
        of the output features extracted form hand-crafted convolution filters.

        """
        import numpy as np
        import tensorflow as tf

        kernel_sizes = [2, 4, 8, 16, 32, 64] if kernel_sizes is None else kernel_sizes

        self.keep_track = 0

        """
        Function to create the hybrid layer consisting of non
        trainable DepthwiseConv1D layers with custom filters.

        Args:

            input_tensor: input tensor
            input_channels : number of input channels, 1 in case of UCR Archive
        """

        conv_list = []

        # for increasing detection filters

        for kernel_size in kernel_sizes:
            filter_ = np.ones(
                shape=(kernel_size, input_channels, 1)
            )  # define the filter weights with the shape corresponding
            # the DepthwiseConv1D layer in keras
            # (kernel_size, input_channels, output_channels)
            indices_ = np.arange(kernel_size)

            filter_[indices_ % 2 == 0] *= -1  # formula of increasing detection filter

            # Create a DepthwiseConv1D layer with non trainable option and no
            # biases and set the filter weights that were calculated in the
            # line above as the initialization

            conv = tf.keras.layers.DepthwiseConv1D(
                kernel_size=kernel_size,
                padding="same",
                use_bias=False,
                depthwise_initializer=tf.keras.initializers.Constant(filter_.tolist()),
                trainable=False,
                name="hybrid-increasse-"
                + str(self.keep_track)
                + "-"
                + str(kernel_size),
            )(input_tensor)

            conv_list.append(conv)  # add the conv layer to the list

            self.keep_track += 1

        # for decreasing detection filters

        for kernel_size in kernel_sizes:
            filter_ = np.ones(
                shape=(kernel_size, input_channels, 1)
            )  # define the filter weights with the shape
            # corresponding the DepthwiseConv1D layer in keras
            # (kernel_size, input_channels, output_channels)
            indices_ = np.arange(kernel_size)

            filter_[indices_ % 2 > 0] *= -1  # formula of decreasing detection filter

            # Create a DepthwiseConv1D layer with non trainable option
            # and no biases and set the filter weights that were
            # calculated in the line above as the initialization

            conv = tf.keras.layers.DepthwiseConv1D(
                kernel_size=kernel_size,
                padding="same",
                use_bias=False,
                depthwise_initializer=tf.keras.initializers.Constant(filter_.tolist()),
                trainable=False,
                name="hybrid-decrease-" + str(self.keep_track) + "-" + str(kernel_size),
            )(input_tensor)

            conv_list.append(conv)  # add the conv layer to the list

            self.keep_track += 1

        # for peak detection filters

        for kernel_size in kernel_sizes[1:]:
            filter_ = np.zeros(
                shape=(kernel_size + kernel_size // 2, input_channels, 1)
            )

            xmesh = np.linspace(start=0, stop=1, num=kernel_size // 4 + 1)[1:].reshape(
                (-1, 1, 1)
            )

            # see utils.custom_filters.py to understand the formulas below

            filter_left = xmesh**2
            filter_right = filter_left[::-1]

            filter_[0 : kernel_size // 4] = -filter_left
            filter_[kernel_size // 4 : kernel_size // 2] = -filter_right
            filter_[kernel_size // 2 : 3 * kernel_size // 4] = 2 * filter_left
            filter_[3 * kernel_size // 4 : kernel_size] = 2 * filter_right
            filter_[kernel_size : 5 * kernel_size // 4] = -filter_left
            filter_[5 * kernel_size // 4 :] = -filter_right

            # Create a DepthwiseConv1D layer with non trainable option and
            # no biases and set the filter weights that were
            # calculated in the line above as the initialization

            conv = tf.keras.layers.DepthwiseConv1D(
                kernel_size=kernel_size + kernel_size // 2,
                padding="same",
                use_bias=False,
                depthwise_initializer=tf.keras.initializers.Constant(filter_.tolist()),
                trainable=False,
                name="hybrid-peeks-" + str(self.keep_track) + "-" + str(kernel_size),
            )(input_tensor)

            conv_list.append(conv)  # add the conv layer to the list

            self.keep_track += 1

        hybrid_layer = tf.keras.layers.Concatenate(axis=2)(
            conv_list
        )  # concantenate all convolution layers
        hybrid_layer = tf.keras.layers.Activation(activation="relu")(
            hybrid_layer
        )  # apply activation ReLU

        return hybrid_layer

    def _inception_module(
        self,
        input_tensor,
        dilation_rate,
        stride=1,
        activation="linear",
        use_custom_filters=True,
        use_multiplexing=True,
    ):
        import tensorflow as tf

        input_inception = input_tensor

        if not use_multiplexing:
            n_convs = 1
            n_filters = self.n_filters * 3

        else:
            n_convs = 3
            n_filters = self.n_filters

        kernel_size_s = [self.kernel_size // (2**i) for i in range(n_convs)]

        conv_list = []

        for i in range(len(kernel_size_s)):
            conv_list.append(
                tf.keras.layers.SeparableConv1D(
                    filters=n_filters,
                    kernel_size=kernel_size_s[i],
                    strides=stride,
                    padding="same",
                    dilation_rate=dilation_rate,
                    activation=activation,
                    use_bias=False,
                )(input_inception)
            )

        if use_custom_filters:
            hybrid_layer = self.hybrid_layer(
                input_tensor=input_tensor, input_channels=input_tensor.shape[-1]
            )
            conv_list.append(hybrid_layer)

        if len(conv_list) > 1:
            x = tf.keras.layers.Concatenate(axis=2)(conv_list)
        else:
            x = conv_list[0]

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation="relu")(x)

        return x

    def _fcn_module(
        self,
        input_tensor,
        kernel_size=20,
        dilation_rate=2,
        n_filters=32,
        stride=1,
        activation="relu",
    ):
        import tensorflow as tf

        x = tf.keras.layers.SeparableConv1D(
            filters=n_filters,
            kernel_size=kernel_size,
            padding="same",
            strides=stride,
            dilation_rate=dilation_rate,
            use_bias=False,
        )(input_tensor)

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation=activation)(x)

        return x

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

        self._build_model()
        self.model.summary()

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
