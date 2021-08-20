# IMPORTS

import tensorflow.keras as kr
from tensorflow.python.keras.layers.convolutional import Conv2D, DepthwiseConv2D
from tensorflow.python.keras.layers.normalization_v2 import BatchNormalization


# CLASSES


class _Identity(kr.layers.Layer):
    """
    generate the Identity layer to be used as appropriate.

    Parameters
    ----------
    name: str
        layer name
    """

    def __init__(self, name):
        assert isinstance(name, str), "'name' must be a str."
        super(_Identity, self).__init__(name=name)
        self.trainable = False

    def __call__(self, inputs):
        """
        handle a call to the class object and return the concatenated layers.

        Parameters
        ----------
        inputs: kr.layers.Layer
            the Keras layer to be used as input.

        Returns
        -------
        block: Keras.layers.Layer
            the layer being the concatenation of the 2D convolutions with, optionally, batch normalization
            and the application of the activation function.
        """
        return inputs


class _LayerPipe:
    """
    generate a pipeline of layers to be called sequentially.

    Parameters
    ----------
    name: str
        the Pipe layer name.

    layers: list
        a list of Keras.layers.Layer objects to be called sequentially once the Pipe is called.

    share_inputs: bool
        True means that the input data are provided to all the entry points of the pipe.

    Notes
    -----
    "share_inputs" affects only he input layer of the Pipe.

    If the input to the Pipe is a Keras.Layer, then "share_inputs" ensures that it is provided to all the elements
    of the first layer of the Pipe. This makes sense if the same input is the start of multiple pipe branches.

    Conversely, if the input to the Pipe is a list (of layers), then, "share_inputs=True" indicates that all the
    elements of the list must be provided to each branch of the Pipe. Otherwise, setting "share_inputs=False" will
    result in having each input element feeding one single branch of the Pipe. This means that when
    "share_inputs=False" the number of inputs must be the same as the number of elements in the first layer of the
    Pipe.
    """

    def __init__(self, layers, name, share_inputs=True):

        # check the entries
        def is_pipeable(obj):
            """
            control whether the object is a keras Layer instance or a list of layers.
            """
            if isinstance(obj, (kr.layers.Layer, _LayerPipe)):
                return True
            elif isinstance(obj, (list, tuple)):
                return all([is_pipeable(i) for i in obj])
            return False

        assert isinstance(name, str), "'name' must be an str object."
        txt = "'layers' must be a list or tuple of keras.layer.Layer objects."
        assert is_pipeable(layers), txt
        assert isinstance(share_inputs, bool), "'share_inputs' must be True or False."

        # add the layers
        def renamer(obj, name):
            """
            add the provided name to the layers name.
            """
            if isinstance(obj, kr.layers.Layer):
                obj.name = "{} - {}".format(name, obj.name)
                return obj
            elif isinstance(obj, (list, tuple)):
                return [renamer(i, name) for i in obj]

        self.layers = renamer(layers)

        # add the sharing options
        self.share_inputs = share_inputs

    def __call__(self, inputs):
        """
        handle a call to the class object and return the concatenated layers.

        Parameters
        ----------
        inputs: kr.layers.Layer or list or tuple
            the Keras layer to be used as input.

        Returns
        -------
        block: Keras.layers.Layer
            the layer being the concatenation of the layers provided according to the Pipe share_inputs and recursive
            options.
        """

        def recaller(obj, arg):
            """
            recursive function to allow the call of nested layers.
            """
            if isinstance(obj, (list, tuple)):
                return [recaller(inner, arg) for inner in obj]
            else:
                return obj(arg)

        # check the shape of the inputs
        if self.share_inputs:
            if isinstance(self.layers[0], kr.layers.Layer):
                x = [inputs]
            else:
                x = [inputs for _ in range(len(self.layers[0]))]
        else:
            x = inputs

        # resolve the pipe

        for i, layer in enumerate(self.layers):
            if isinstance(layer, (list, tuple)):
                if i == 0 and self.share_inputs:
                    x = [recaller(l, v) for l, v in zip(layer, x)]
                else:
                    x = recaller(layer, x)
            else:
                x = layer(x)
        return x


class _Stem(_LayerPipe):
    """
    generate the Stem Pipe of the Semantic branch being part of the BiSeNet V2.

    Parameters
    ----------
    kernel_size: int
        the basic size of the convolutional kernel for each level of the block.

    channels: int
        the basic number of channels of each block.

    name: str
        layer name
    """

    def __init__(self, kernel_size, channels, name):

        # check the entries
        assert isinstance(kernel_size, int), "'kernel_size' must be an int object."
        assert isinstance(channels, int), "'channels' must be an int object."
        assert isinstance(name, str), "'name' must be an str object."

        # create the blocks
        layers = [
            _LayerPipe(
                layers=[
                    kr.layers.Conv2D(
                        kernel_size=kernel_size,
                        filters=channels,
                        stride=2,
                        name="{}x{}Conv2D (stride {})".format(
                            kernel_size, kernel_size, 2
                        ),
                        padding="same",
                    ),
                    kr.layers.BatchNormalization(name="BatchNorm"),
                    kr.layersReLU(name="ReLU"),
                ],
                name="Layer1",
            ),
            [
                _LayerPipe(
                    layers=[
                        kr.layers.Conv2D(
                            kernel_size=1,
                            filters=channels,
                            stride=1,
                            name="{}x{}Conv2D (stride {})".format(1, 1, 1),
                            padding="same",
                        ),
                        kr.layers.BatchNormalization(name="BatchNorm"),
                        kr.layersReLU(name="ReLU"),
                        kr.layers.Conv2D(
                            kernel_size=kernel_size,
                            filters=channels,
                            stride=2,
                            name="{}x{}Conv2D (stride {})".format(
                                kernel_size, kernel_size, 2
                            ),
                            padding="same",
                        ),
                        kr.layers.BatchNormalization(name="BatchNorm"),
                        kr.layersReLU(name="ReLU"),
                    ],
                    name="Layer2 - Left Branch",
                ),
                kr.layers.MaxPooling2D(
                    pool_size=kernel_size,
                    strides=2,
                    padding="same",
                    name="Layer2 - Right Branch - MaxPool",
                ),
            ],
            kr.layers.Concatenate(name="Layer3 - Concatenation"),
            _LayerPipe(
                layers=[
                    kr.layers.Conv2D(
                        kernel_size=kernel_size,
                        filters=channels,
                        stride=1,
                        name="{}x{}Conv2D (stride {})".format(
                            kernel_size, kernel_size, 1
                        ),
                        padding="same",
                    ),
                    kr.layers.BatchNormalization(name="BatchNorm"),
                    kr.layersReLU(name="ReLU"),
                ],
                name="Layer4",
            ),
        ]

        # initialize
        super(_Stem, self).__init__(layers, name)


class _GatherExpansion(_LayerPipe):
    """
    generate the Gather and Expansion layer(s) of the Semantic branch being part of the BiSeNet V2.

    Parameters
    ----------
    kernel_size: int
        the basic size of the convolutional kernel for each level of the block.

    channels: int
        the basic number of channels of each block.

    stride: int
        the strides to be applied. (1 or 2).

    expansion: int
        the expansion factor to be used during the Depthwise Convolution to vary the number of
        channels.

    name: str
        layer name
    """

    def __init__(self, kernel_size, channels, stride, expansion, name):

        # check the entries
        assert isinstance(kernel_size, int), "'kernel_size' must be an int object."
        assert isinstance(channels, int), "'channels' must be an int object."
        assert isinstance(stride, int), "'stride' must be an int object."
        assert stride in [1, 2], "'stride' can be only 1 or 2."
        assert isinstance(expansion, int), "'expansion' must be an int object."
        assert isinstance(name, str), "'name' must be an str object."

        # here the layers are put in lists indicating how to
        # process them by the __call__ method.
        left_branch_layers = [
            _LayerPipe(
                layers=[
                    kr.layers.Conv2D(
                        kernel_size=kernel_size,
                        filters=channels,
                        stride=1,
                        name="{}x{}Conv2D (stride {})".format(
                            kernel_size, kernel_size, 1
                        ),
                        padding="same",
                    ),
                    kr.layers.BatchNormalization(name="BatchNorm"),
                    kr.layersReLU(name="ReLU"),
                ],
                name="Layer1",
            ),
            _LayerPipe(
                layers=[
                    kr.layers.DepthwiseConv2D(
                        kernel_size=kernel_size,
                        stride=stride,
                        depth_multiplier=expansion,
                        name="{}x{}DepthwiseConv2D (stride {}, exp {})".format(
                            kernel_size, kernel_size, stride, expansion
                        ),
                        padding="same",
                    ),
                    kr.layers.BatchNormalization(name="BatchNorm"),
                ],
                name="Layer2",
            ),
        ]
        n_left = 2

        if stride == 2:

            left_branch_layers += [
                _LayerPipe(
                    layers=[
                        kr.layers.DepthwiseConv2D(
                            kernel_size=kernel_size,
                            stride=1,
                            depth_multiplier=expansion,
                            name="{}x{}DepthwiseConv2D (stride {}, exp {})".format(
                                kernel_size, kernel_size, 1, expansion
                            ),
                            padding="same",
                        ),
                        kr.layers.BatchNormalization(name="BatchNorm"),
                    ],
                    name="Layer3",
                )
            ]
            n_left += 1
            right_branch_layers = [
                _LayerPipe(
                    layers=[
                        kr.layers.DepthwiseConv2D(
                            kernel_size=kernel_size,
                            stride=stride,
                            depth_multiplier=expansion,
                            name="{}x{}DepthwiseConv2D (stride {}, exp {})".format(
                                kernel_size, kernel_size, stride, expansion
                            ),
                            padding="same",
                        ),
                        kr.layers.BatchNormalization(name="BatchNorm"),
                    ],
                    name="Layer1",
                ),
                _LayerPipe(
                    layers=[
                        kr.layers.Conv2D(
                            kernel_size=1,
                            stride=1,
                            filters=channels,
                            name="{}x{}Conv2D (stride {})".format(1, 1, 1),
                            padding="same",
                        ),
                        kr.layers.BatchNormalization(name="BatchNorm"),
                    ],
                    name="Layer2",
                ),
            ]
        else:

            right_branch_layers = [_Identity(name="Layer1 - Identity")]

        left_branch_layers += [
            _LayerPipe(
                layers=[
                    kr.layers.Conv2D(
                        kernel_size=1,
                        stride=1,
                        filters=channels,
                        name="{}x{}Conv2D (stride {})".format(1, 1, 1),
                        padding="same",
                    ),
                    kr.layers.BatchNormalization(name="BatchNorm"),
                ],
                name="Layer{}".format(n_left + 1),
            )
        ]

        layers = [
            [
                _LayerPipe(left_branch_layers, "Left Branch"),
                _LayerPipe(right_branch_layers, "Right Branch"),
            ],
            kr.layers.Add(name="Sum"),
            kr.layersReLU(name="ReLU"),
        ]

        # initialization
        super(_GatherExpansion, self).__init__(layers, name)


class _ContextEmbedding(_LayerPipe):
    """
    generate the Context Embedding layer(s) of the Semantic branch being part of the BiSeNet V2.

    Parameters
    ----------
    kernel_size: int
        the basic size of the convolutional kernel for each level of the block.

    channels: int
        the basic number of channels of each block.

    name: str
        the name of the layer
    """

    def __init__(self, kernel_size, channels, name):

        # check the entries
        assert isinstance(kernel_size, int), "'kernel_size' must be an int object."
        assert isinstance(channels, int), "'channels' must be an int object."
        assert isinstance(name, str), "'name' must be an str object."

        # create the blocks
        layers = [
            [
                _LayerPipe(
                    layers=[
                        _LayerPipe(
                            layers=[
                                kr.layers.GlobalAvgPool2D(name="GAPool"),
                                kr.layers.BatchNormalization(name="BatchNorm"),
                            ],
                            name="Layer1",
                        ),
                        _LayerPipe(
                            layers=[
                                kr.layers.Conv2D(
                                    kernel_size=1,
                                    stride=1,
                                    filters=channels,
                                    name="{}x{}Conv2D (stride {})".format(1, 1, 1),
                                    padding="same",
                                ),
                                kr.layers.BatchNormalization(name="BatchNorm"),
                                kr.layers.ReLU(name="ReLU"),
                            ],
                            name="Layer2",
                        ),
                    ],
                    name="Left Branch",
                ),
                _Identity(name="Right Branch - Layer1 - Identity"),
            ],
            kr.layers.Add(name="Layer3 - Add"),
            kr.layers.Conv2D(
                kernel_size=kernel_size,
                stride=1,
                filters=channels,
                name="Layer4 - {}x{}Conv2D (stride {})".format(
                    kernel_size, kernel_size, 1
                ),
                padding="same",
            ),
        ]

        # init
        super(_ContextEmbedding, self).__init__(layers, name)


class _DetailBranch(_LayerPipe):
    """
    generate the Detail Branch of the BiSeNet V2 by concatenating a set of convolutional blocks.

    Parameters
    ----------
    kernel_size: list or tuple, optional
        the size of the convolutional kernel for each level of the block.

    channels: int
        the basic number of channels of each block.

    Notes
    -----
    All the passed parameters should be lists or tuple having len equal to the number of desired
    convolutional layers to be included.
    """

    def __init__(self, kernel_size, channels):

        # check the entries
        assert isinstance(kernel_size, int), "'kernel_size' must be an int object."
        assert isinstance(channels, int), "'channels' must be an int object."

        # create the layers
        layers = [
            _LayerPipe(
                layers=[
                    kr.layers.Conv2D(
                        kernel_size=kernel_size,
                        stride=2,
                        filters=channels,
                        name="{}x{}Conv2D (stride {})".format(
                            kernel_size, kernel_size, 2
                        ),
                        padding="same",
                    ),
                    kr.layers.Conv2D(
                        kernel_size=kernel_size,
                        stride=1,
                        filters=channels,
                        name="{}x{}Conv2D (stride {})".format(
                            kernel_size, kernel_size, 1
                        ),
                        padding="same",
                    ),
                ],
                name="Stage1",
            ),
            _LayerPipe(
                layers=[
                    kr.layers.Conv2D(
                        kernel_size=kernel_size,
                        stride=2,
                        filters=channels,
                        name="{}x{}Conv2D (stride {})".format(
                            kernel_size, kernel_size, 2
                        ),
                        padding="same",
                    ),
                    kr.layers.Conv2D(
                        kernel_size=kernel_size,
                        stride=1,
                        filters=channels,
                        name="{}x{}Conv2D (stride {})".format(
                            kernel_size, kernel_size, 1
                        ),
                        padding="same",
                    ),
                    kr.layers.Conv2D(
                        kernel_size=kernel_size,
                        stride=1,
                        filters=channels,
                        name="{}x{}Conv2D (stride {})".format(
                            kernel_size, kernel_size, 1
                        ),
                        padding="same",
                    ),
                ],
                name="Stage2",
            ),
            _LayerPipe(
                layers=[
                    kr.layers.Conv2D(
                        kernel_size=kernel_size,
                        stride=2,
                        filters=channels * 2,
                        name="{}x{}Conv2D (stride {})".format(
                            kernel_size, kernel_size, 2
                        ),
                        padding="same",
                    ),
                    kr.layers.Conv2D(
                        kernel_size=kernel_size,
                        stride=1,
                        filters=channels * 2,
                        name="{}x{}Conv2D (stride {})".format(
                            kernel_size, kernel_size, 1
                        ),
                        padding="same",
                    ),
                    kr.layers.Conv2D(
                        kernel_size=kernel_size,
                        stride=1,
                        filters=channels * 2,
                        name="{}x{}Conv2D (stride {})".format(
                            kernel_size, kernel_size, 1
                        ),
                        padding="same",
                    ),
                ],
                name="Stage3",
            ),
        ]

        # init
        super(_DetailBranch, self).__init__(layers, "Detail Branch")


class _SemanticBranch(_LayerPipe):
    """
    generate the Semantic Branch of the BiSeNet V2 by concatenating a set of convolutional blocks.

    Parameters
    ----------
    kernel_size: list or tuple, optional
        the size of the convolutional kernel for each level of the block.

    channels: int
        the basic number of channels of each block.

    Notes
    -----
    All the passed parameters should be lists or tuple having len equal to the number of desired
    convolutional layers to be included.
    """

    def __init__(self, kernel_size, channels):

        # check the entries
        assert isinstance(kernel_size, int), "'kernel_size' must be an int object."
        assert isinstance(channels, int), "'channels' must be an int object."

        # create the blocks
        layers = [
            _Stem(kernel_size, int(channels / 4), "Stage 1&2 - Stem"),
            _LayerPipe(
                layers=[
                    _GatherExpansion(kernel_size, int(channels / 2), 2, 6, "GE1"),
                    _GatherExpansion(kernel_size, int(channels / 2), 1, 6, "GE2"),
                ],
                name="Stage3",
            ),
            _LayerPipe(
                layers=[
                    _GatherExpansion(kernel_size, channels, 2, 6, "GE3"),
                    _GatherExpansion(kernel_size, channels, 1, 6, "GE4"),
                ],
                name="Stage4",
            ),
            _LayerPipe(
                layers=[
                    _GatherExpansion(kernel_size, channels * 2, 2, 6, "GE5"),
                    _GatherExpansion(kernel_size, channels * 2, 1, 6, "GE6"),
                    _ContextEmbedding(kernel_size, channels, "CE"),
                ],
                name="Stage5",
            ),
        ]

        # initialize
        super(_SemanticBranch, self).__init__(layers, "Semantic branch")


class _BilateralGuidedAggregation(_LayerPipe):
    """
    generate the Semantic Branch of the BiSeNet V2 by concatenating a set of convolutional blocks.

    Parameters
    ----------
    kernel_size: list or tuple, optional
        the size of the convolutional kernel for each level of the block.

    channels: int
        the basic number of channels of each block.

    Notes
    -----
    All the passed parameters should be lists or tuple having len equal to the number of desired
    convolutional layers to be included.
    """

    def __init__(self, kernel_size, channels):

        # check the entries
        assert isinstance(kernel_size, int), "'kernel_size' must be an int object."
        assert isinstance(channels, int), "'channels' must be an int object."

        # make the branches
        layers = [
            [
                _LayerPipe(
                    name="Left Branch",
                    share_inputs=False,
                    layers=[
                        [  # Detail part
                            kr.layers.DepthwiseConv2D(
                                kernel_size=kernel_size,
                                stride=1,
                                depth_multiplier=1,
                                name="{}x{}DepthwiseConv2D (stride {}, exp {})".format(
                                    kernel_size, kernel_size, 1, 1
                                ),
                                padding="same",
                            ),
                            kr.layers.BatchNormalization(name="BatchNorm"),
                            kr.layers.DepthwiseConv2D(
                                kernel_size=1,
                                stride=1,
                                filters=channels,
                                name="{}x{}Conv2D (stride {})".format(1, 1, 1),
                                padding="same",
                            ),
                        ],
                        [  # Semantic part
                            kr.layers.Conv2D(
                                kernel_size=kernel_size,
                                stride=1,
                                filters=channels,
                                name="{}x{}Conv2D (stride {})".format(
                                    kernel_size, kernel_size, 1
                                ),
                                padding="same",
                            ),
                            kr.layers.BatchNormalization(name="BatchNorm"),
                            kr.layers.UpSampling2D(
                                size=4, interpolation="bilinear", name="4xUpsampling"
                            ),
                        ],
                        kr.layers.Multiply(name="Multiplication"),
                    ],
                ),
                _LayerPipe(
                    name="Right Branch",
                    share_inputs=False,
                    layers=[
                        [  # detail part
                            kr.layers.Conv2D(
                                kernel_size=kernel_size,
                                stride=2,
                                filters=channels,
                                name="{}x{}Conv2D (stride {})".format(
                                    kernel_size, kernel_size, 1
                                ),
                                padding="same",
                            ),
                            kr.layers.BatchNormalization(name="BatchNorm"),
                            kr.layers.GlobalAvgPool2D(
                                kernel_size=2, stride=2, name="GAPool"
                            ),
                        ],
                        [  # semantic part
                            kr.layers.Conv2D(
                                kernel_size=kernel_size,
                                stride=1,
                                filters=channels,
                                name="{}x{}Conv2D (stride {})".format(
                                    kernel_size, kernel_size, 1
                                ),
                                padding="same",
                            ),
                            kr.layers.BatchNormalization(name="BatchNorm"),
                            kr.layers.UpSampling2D(
                                size=4, interpolation="bilinear", name="4xUpsampling"
                            ),
                        ],
                        kr.layers.Multiply(name="Multiplication"),
                    ],
                ),
            ],
            kr.layers.Add(name="Sum"),
            kr.layers.Conv2D(
                kernel_size=kernel_size,
                stride=1,
                filters=channels,
                name="{}x{}Conv2D (stride {})".format(kernel_size, kernel_size, 1),
                padding="same",
            ),
            kr.layers.BatchNormalization(name="BatchNorm"),
        ]

        # initialize
        super(_BilateralGuidedAggregation, self).__init__(layers, "Guided Aggregation")


class BiSeNet2(kr.Model):
    def __init__(self, kernel_size=3, **kwargs):
        """
        generate a BiSeNet (V2) model.

        Parameters
        ----------

        kernel_size: list or tuple, optional
            the size of the convolutional kernel for each level of the block.

        kwargs: any
            additional parameters to be passed to the convolutional layers.

        Returns
        -------
        block: Keras.Model
            the BiSeNet2 model.

        Notes
        -----
        As described in the paper, the kernel size is kept constant across all the blocks and stages.
        """

        self._kernel_size = kernel_size

        # build the model
        x = kr.layers.Input(name="Input")  # input
        detail_block = _DetailBranch(self._kernel_size)(x)  # detail block
        semantic_block = _SemanticBranch(self._kernel_size)(x)  # semanting block
