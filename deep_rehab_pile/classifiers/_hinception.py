"""Hybrid Inception classifier."""

__all__ = ["H_Inception_CLASSIFIER"]

import tensorflow as tf
from aeon.classification.deep_learning import IndividualInceptionClassifier

from deep_rehab_pile.classifiers.base import BASE_CLASSIFIER


class H_Inception_CLASSIFIER(BASE_CLASSIFIER):
    """Define Hybrid Inception Classifier.

    This class implements a Hybrid Inception network-based classifier
    proposed in [1]_ and made hybrid with hand-crafted filters
    in [2]_.

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
    depth               : int, default = 6,
        the number of inception modules used
    n_filters          : int or list of int32, default = 32,
        the number of filters used in one inception module, if not a list,
        the same number of filters is used in all inception modules
    n_conv_per_layer   : int or list of int, default = 3,
        the number of convolution layers in each inception module, if not a list,
        the same number of convolution layers is used in all inception modules
    kernel_size         : int or list of int, default = 40,
        the head kernel size used for each inception module, if not a list,
        the same is used in all inception modules
    use_max_pooling     : bool or list of bool, default = True,
        conditioning whether or not to use max pooling layer
        in inception modules,if not a list,
        the same is used in all inception modules
    max_pool_size       : int or list of int, default = 3,
        the size of the max pooling layer, if not a list,
        the same is used in all inception modules
    strides             : int or list of int, default = 1,
        the strides of kernels in convolution layers for
        each inception module, if not a list,
        the same is used in all inception modules
    dilation_rate       : int or list of int, default = 1,
        the dilation rate of convolutions in each inception module, if not a list,
        the same is used in all inception modules
    padding             : str or list of str, default = "same",
        the type of padding used for convoltuon for each
        inception module, if not a list,
        the same is used in all inception modules
    activation          : str or list of str, default = "relu",
        the activation function used in each inception module, if not a list,
        the same is used in all inception modules
    use_bias            : bool or list of bool, default = False,
        conditioning whether or not convolutions should
        use bias values in each inception
        module, if not a list,
        the same is used in all inception modules
    use_residual        : bool, default = True,
        condition whether or not to use residual connections all over Inception
    use_bottleneck      : bool, default = True,
        confition whether or not to use bottlenecks all over Inception
    bottleneck_size     : int, default = 32,
        the bottleneck size in case use_bottleneck = True

    References
    ----------
    .. [1] Ismail Fawaz, Hassan, et al. "Inceptiontime: Finding
    alexnet for time series classification." Data Mining and
    Knowledge Discovery 34.6 (2020): 1936-1962.
    .. [2] Ismail-Fawaz, Ali, et al. "Deep Learning For Time
    Series Classification Using New Hand-Crafted Convolution
    Filters" IEEE International Conference on Big Data
    (Big Data). IEEE, 2022.
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
        depth: int = 6,
        n_filters: int = 32,
        n_conv_per_layer: int = 3,
        kernel_size: int = 40,
        use_max_pooling: bool = True,
        max_pool_size: int = 3,
        strides: int = 1,
        dilation_rate: int = 1,
        padding: str = "same",
        activation: str = "relu",
        use_bias: bool = False,
        use_residual: bool = True,
        use_bottleneck: bool = True,
        bottleneck_size: int = 32,
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

        self.depth = depth
        self.n_filters = n_filters
        self.n_conv_per_layer = n_conv_per_layer
        self.kernel_size = kernel_size
        self.use_max_pooling = use_max_pooling
        self.max_pool_size = max_pool_size
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.padding = padding
        self.activation = activation
        self.use_bias = use_bias
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.bottleneck_size = bottleneck_size

        self.classifier = IndividualInceptionClassifier(
            depth=self.depth,
            n_filters=self.n_filters,
            n_conv_per_layer=self.n_conv_per_layer,
            kernel_size=self.kernel_size,
            use_max_pooling=self.use_max_pooling,
            max_pool_size=self.max_pool_size,
            strides=self.strides,
            dilation_rate=self.dilation_rate,
            padding=self.padding,
            activation=self.activation,
            use_bias=self.use_bias,
            use_residual=self.use_residual,
            use_bottleneck=self.use_bottleneck,
            bottleneck_size=self.bottleneck_size,
            use_custom_filters=True,
            file_path=self.output_dir,
            save_best_model=True,
            save_init_model=True,
            best_file_name=self.best_file_name,
            init_file_name=self.init_file_name,
            batch_size=self.batch_size,
            n_epochs=self.epochs,
            verbose=1,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=self.output_dir + self.best_file_name + ".keras",
                    monitor="loss",
                    save_best_only=True,
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="loss", factor=0.5, patience=50, min_lr=0.0001
                ),
            ],
        )
