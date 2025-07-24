"""FCN regressor."""

__all__ = ["FCN_REGRESSOR"]


import tensorflow as tf
from aeon.regression.deep_learning import FCNRegressor

from deep_rehab_pile.regressors.base import BASE_REGRESSOR


class FCN_REGRESSOR(BASE_REGRESSOR):
    """Define Fully Convolutional Network (FCN) Regressor.

    This class implements a fully convolutional network-based regressor
    adapted from [1]_.

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
    n_layers : int, default = 3
        Number of convolution layers.
    n_filters : int or list of int, default = [128,256,128]
        Number of filters used in convolution layers.
    kernel_size : int or list of int, default = [8,5,3]
        Size of convolution kernel.
    dilation_rate : int or list of int, default = 1
        The dilation rate for convolution.
    strides : int or list of int, default = 1
        The strides of the convolution filter.
    padding : str or list of str, default = "same"
        The type of padding used for convolution.
    activation : str or list of str, default = "relu"
        Activation used after the convolution.
    use_bias : bool or list of bool, default = True
        Whether or not ot use bias in convolution.

    References
    ----------
    .. [1] Wang, Zhiguang, Weizhong Yan, and Tim Oates. "Time series
    classification from scratch with deep neural networks: A strong baseline."
    2017 International joint conference on neural networks (IJCNN). IEEE, 2017.
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
        n_layers: int = 3,
        n_filters: list = None,
        kernel_size: list = None,
        dilation_rate: int = 1,
        strides: int = 1,
        padding: str = "same",
        activation: str = "relu",
        use_bias: bool = True,
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

        self.n_layers = n_layers
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.use_bias = use_bias

        self.regressor = FCNRegressor(
            n_layers=self.n_layers,
            n_filters=self.n_filters,
            kernel_size=self.kernel_size,
            dilation_rate=self.dilation_rate,
            strides=self.strides,
            padding=self.padding,
            activation=self.activation,
            use_bias=self.use_bias,
            file_path=self.output_dir,
            save_best_model=True,
            save_init_model=True,
            best_file_name=self.best_file_name,
            init_file_name=self.init_file_name,
            batch_size=self.batch_size,
            n_epochs=self.epochs,
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
            verbose=1,
        )
