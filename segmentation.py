# IMPORTS

import tensorflow.keras as kr
import tensorflow as tf


# METHODS


def conv2d(
    kernel_size=3,
    output_channels=64,
    stride=1,
    use_bias=False,
    padding="same",
    **kwargs
):
    """
    wrapper of the Keras.layers.Conv2D Layer.

    Parameters
    ----------
    kernel_size: int, list, tuple
        the size of the kernel.
        If a list or tuple is given, its dimensions will correspond to the Height and Width kernels.
        If a single integer is provided, it is shared across dimensions.

    output_channels: int
        the number of output channels of the convolved layer.

    stride: int
        the stride to be applied to the convolutional layer.

    use_bias: bool
        should a bias array be added to the weights.

    padding: str
        the padding style. The available options are "same" (default) or "valid".

    kwargs: any
        additional parameters passed to the Conv2D keras class.


    Returns
    -------
    layer: Keras.layer.Conv2D
        the convolutional layer correpsonding to the input data.
    """

    # check the entries
    txt = "'kernel_size must be an int or a list/tuple of int with len = 2."
    if isinstance(kernel_size, (tuple, list)):
        assert len(kernel_size) == 2, txt
        assert all([isinstance(i, int) for i in kernel_size]), txt
    else:
        assert isinstance(kernel_size, int), txt
        kernel_size = [kernel_size, kernel_size]
    txt = lambda x, y: "{} must be of class {}.".format(x, y)
    assert isinstance(output_channels, int), txt("output_channels", int)
    assert isinstance(stride, int), txt("stride", int)
    assert isinstance(use_bias, bool), txt("use_bias", bool)
    assert isinstance(padding, str), txt("padding", str)
    valid_padding = ["same", "valid"]
    txt = "Available 'padding' values are {}.".format(valid_padding)
    assert padding in valid_padding, txt

    # check the name property
    if any([i == "name" for i in kwargs]):
        name = kwargs.pop("name")
        name = name.replace(" ", "")
    else:
        name = "{}x{}x{}x{}Conv2D"
        name = name.format(kernel_size[0], kernel_size[1], output_channels, stride)
    return kr.layers.Conv2D(
        kernel_size=kernel_size,
        strides=stride,
        filters=output_channels,
        use_bias=use_bias,
        padding=padding,
        name=name,
        **kwargs
    )


def conv2d_bn_relu(
    kernel_size=3,
    output_channels=64,
    stride=1,
    batch_normalization=True,
    relu_activation=True,
    name="",
    **kwargs
):
    """
    return a _LayerPipe object serializing a convolutional layer optionally followed by batch normalization
    and ReLU activation.

    Parameters
    ----------
    kernel_size: int, list, tuple
        the size of the kernel.
        If a list or tuple is given, its dimensions will correspond to the Height and Width kernels.
        If a single integer is provided, it is shared across dimensions.

    output_channels: int
        the number of output channels of the convolved layer.

    stride: int
        the stride to be applied to the convolutional layer.

    batch_normalization: bool
        should the output of the convolution be batch_normalized?

    relu_activation: bool
        should the output of the convolution pass through a ReLU operator?

    name: str
        a name for the Pipe.

    kwargs: any
        additional parameters passed to the Conv2D keras class.


    Returns
    -------
    layer: Keras.layer.Conv2D
        the convolutional layer correpsonding to the input data.
    """

    # parameters check
    txt = "'{}' must be True or False."
    assert isinstance(batch_normalization, bool), txt.format("batch_normalization")
    assert isinstance(relu_activation, bool), txt.format("relu_activation")
    assert isinstance(name, str), "'name' must be a string."

    # get the layers
    layers = [
        conv2d(
            kernel_size=kernel_size,
            output_channels=output_channels,
            stride=stride,
            **kwargs
        )
    ]
    if batch_normalization:
        layers += [kr.layers.BatchNormalization(name="BatchNorm")]
    if relu_activation:
        layers += [kr.layers.ReLU(name="ReLU")]

    # make the pipe
    return _LayerPipe(layers=layers, name=name.replace(" ", ""))


def depthwise_conv2_bn(
    kernel_size=3, stride=1, multiplier=6, use_bias=False, padding="same", **kwargs
):
    """
    return a Depthwise Convolutional 2D layer followed by batch normalization.

    Parameters
    ----------

    kernel_size: int, list, tuple
        the size of the kernel.
        If a list or tuple is given, its dimensions will correspond to the Height and Width kernels.
        If a single integer is provided, it is shared across dimensions.

    multiplier: int
        the filter depth multiplier.

    stride: int
        the stride to be applied to the convolutional layer.

    use_bias: bool
        should a bias array be added to the weights.

    padding: str
        the padding style. The available options are "same" (default) or "valid".

    kwargs: any
        additional parameters passed to the Conv2D keras class.

    Returns
    -------
    layer: _LayerPipe
        the pipe of depthwise convolutional layer followed by batch normalization.
    """

    # check the entries
    txt = "'kernel_size must be an int or a list/tuple of int with len = 2."
    if isinstance(kernel_size, (tuple, list)):
        assert len(kernel_size) == 2, txt
        assert all([isinstance(i, int) for i in kernel_size]), txt
    else:
        assert isinstance(kernel_size, int), txt
        kernel_size = [kernel_size, kernel_size]
    txt = lambda x, y: "{} must be of class {}.".format(x, y)
    assert isinstance(stride, int), txt("stride", int)
    assert isinstance(use_bias, bool), txt("use_bias", bool)
    assert isinstance(padding, str), txt("padding", str)
    valid_padding = ["same", "valid"]
    txt = "Available 'padding' values are {}.".format(valid_padding)
    assert padding in valid_padding, txt

    # check the name property
    if any([i == "name" for i in kwargs]):
        name = kwargs.pop("name")
        name = name.replace(" ", "")
    else:
        name = "{}x{}x{}x{}DepthwiseConv2D)"
        name = name.format(kernel_size[0], kernel_size[1], multiplier, stride)

    return _LayerPipe(
        layers=[
            kr.layers.DepthwiseConv2D(
                kernel_size=kernel_size,
                strides=stride,
                depth_multiplier=multiplier,
                name=name,
                padding=padding,
                use_bias=use_bias,
                **kwargs
            ),
            kr.layers.BatchNormalization(name="BatchNorm"),
        ],
        name=name,
    )


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

    def build(self, input_shape):  # Create the state of the layer (weights)
        w_init = tf.ones_initializer()
        self.w = tf.Variable(initial_value=w_init(shape=input_shape), trainable=False)

    def call(self, inputs):
        """
        handle a call to the class object and return the concatenated layers.

        Parameters
        ----------
        inputs: kr.layers.Layer
            the Keras layer to be used as input.

        Returns
        -------
        block: Keras.layers.Layer
            passed layer.
        """
        return tf.matmul(inputs, self.w)

    @property
    def trainable(self):
        """
        Identity Layer cannot be trained.
        """
        return False


class _LayerPipe(object):
    """
    generate a pipeline of layers to be called sequentially.

    Parameters
    ----------
    layers: list
        a list of Keras.layers.Layer objects to be called sequentially once the Pipe is called.

    name: str
        the Pipe layer name.

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

        # add the name
        self._name = name.replace(" ", "")

        # add the layers
        def renamer(obj, name):
            """
            add the provided name to the layers name.
            """
            if isinstance(obj, (kr.layers.Layer)):
                obj._name = "{}-{}".format(name, obj._name)
                return obj
            elif isinstance(obj, (list, tuple)):
                return [renamer(i, name) for i in obj]
            elif isinstance(obj, _LayerPipe):
                for i in range(len(obj.layers)):
                    obj.layers[i] = renamer(obj.layers[i], name)
                return obj

        self.layers = renamer(layers, self._name)

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

        # check the shape of the inputs
        if self.share_inputs:

            def share_inputs_fun(inputs, first_layer):

                if isinstance(first_layer, kr.layers.Layer):
                    return inputs

                elif isinstance(first_layer, _LayerPipe):
                    return share_inputs_fun(inputs, first_layer.layers[0])

                else:
                    return [inputs for _ in range(len(first_layer))]

            x = share_inputs_fun(inputs, self.layers[0])

        else:
            x = inputs

        # resolve the pipe
        def recaller(obj, arg, share_args=False):
            """
            recursive function to allow the call of nested layers.
            """
            if isinstance(obj, (list, tuple)):
                if share_args:
                    return [recaller(e, arg, share_args) for e in obj]
                else:
                    return [recaller(e, a) for e, a in zip(obj, arg)]

            elif isinstance(obj, _LayerPipe):
                x = arg
                for level in obj.layers:
                    x = recaller(level, x, share_args)
                return x

            else:
                return obj(arg)

        # iterate over the layers
        for i, layer in enumerate(self.layers):
            if i == 0:
                x = recaller(layer, x, self.share_inputs)
            else:
                x = recaller(layer, x, True)
        return x


class _DetailBranch(_LayerPipe):
    """
    generate the Detail Branch of the BiSeNet V2 by concatenating a set of convolutional blocks.

    Parameters
    ----------
    kernel_size: list or tuple, optional
        the size of the convolutional kernel for each level of the block.

    channels: int
        the basic number of channels of each block.

    kwargs: any
        additional parameters passed to the convolutional layers.
    """

    def __init__(self, kernel_size, channels, **kwargs):

        # check the entries
        txt = lambda x, c: "{} must be and object of class {}.".format(x, c)
        assert isinstance(kernel_size, int), txt("kernel_size", int)
        assert isinstance(channels, int), txt("channels", int)

        # create the first 2 stages
        layers = [
            _LayerPipe(
                layers=[
                    conv2d(kernel_size, channels, 2, **kwargs),
                    conv2d(kernel_size, channels, 1, **kwargs),
                ],
                name="Stage1",
            ),
            _LayerPipe(
                layers=[
                    conv2d(kernel_size, channels, 2, **kwargs),
                    conv2d(kernel_size, channels, 1, **kwargs),
                    conv2d(kernel_size, channels, 1, **kwargs),
                ],
                name="Stage2",
            ),
            _LayerPipe(
                layers=[
                    conv2d(kernel_size, channels * 2, 2, **kwargs),
                    conv2d(kernel_size, channels * 2, 1, **kwargs),
                    conv2d(kernel_size, channels * 2, 1, **kwargs),
                ],
                name="Stage3",
            ),
        ]

        # init
        super(_DetailBranch, self).__init__(layers, "Detail Branch")


class _Stem(_LayerPipe):
    """
    generate the Stem Pipe of the Semantic branch being part of the BiSeNet V2.

    Parameters
    ----------
    kernel_size: int
        the basic size of the convolutional kernel for each level of the block.

    channels: int
        the basic number of channels of each block.

    kwargs: any
        additional parameters passed to the convolutional layers.
    """

    def __init__(self, kernel_size, channels, **kwargs):

        # check the entries
        assert isinstance(kernel_size, int), "'kernel_size' must be an int object."
        assert isinstance(channels, int), "'channels' must be an int object."

        # create the blocks
        layers = [
            conv2d_bn_relu(
                kernel_size=kernel_size,
                output_channels=channels,
                stride=2,
                name="Layer1",
                **kwargs
            ),
            [
                _LayerPipe(
                    layers=[
                        conv2d_bn_relu(
                            kernel_size=1,
                            output_channels=channels,
                            stride=1,
                            name="Layer1",
                            **kwargs
                        ),
                        conv2d_bn_relu(
                            kernel_size=kernel_size,
                            output_channels=channels,
                            stride=2,
                            name="Layer2",
                            **kwargs
                        ),
                    ],
                    name="Layer2-LeftBranch",
                ),
                kr.layers.MaxPooling2D(
                    pool_size=kernel_size,
                    strides=2,
                    padding="same",
                    name="Layer2-RightBranch-MaxPool",
                ),
            ],
            kr.layers.Concatenate(name="Layer3-Concatenation"),
            conv2d_bn_relu(
                kernel_size=kernel_size,
                output_channels=channels,
                stride=1,
                name="Layer4",
                **kwargs
            ),
        ]

        # initialize
        super(_Stem, self).__init__(layers, "StemBlock")


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

    kwargs: any
        additional parameters passed to the convolutional layers.
    """

    def __init__(self, kernel_size, channels, stride, **kwargs):

        # check the entries
        assert isinstance(kernel_size, int), "'kernel_size' must be an int object."
        assert isinstance(channels, int), "'channels' must be an int object."
        assert isinstance(stride, int), "'stride' must be an int object."
        assert stride in [1, 2], "'stride' can be only 1 or 2."

        # here the layers are put in lists indicating how to
        # process them by the __call__ method.
        left_branch_layers = [
            conv2d_bn_relu(
                kernel_size=kernel_size,
                output_channels=channels,
                stride=1,
                name="Layer1",
                **kwargs
            ),
            depthwise_conv2_bn(
                kernel_size=kernel_size, stride=stride, multiplier=6, name="Layer2"
            ),
        ]
        n_left = 2

        if stride == 2:

            left_branch_layers += [
                depthwise_conv2_bn(
                    kernel_size=kernel_size, stride=stride, multiplier=6, name="Layer3"
                ),
            ]
            n_left += 1
            right_branch_layers = [
                depthwise_conv2_bn(
                    kernel_size=kernel_size, stride=stride, multiplier=6, name="Layer1"
                ),
                conv2d_bn_relu(
                    kernel_size=1,
                    output_channels=channels,
                    stride=1,
                    name="Layer2",
                    **kwargs
                ),
            ]
        else:

            right_branch_layers = [_Identity(name="Layer1-Identity")]

        left_branch_layers += [
            conv2d_bn_relu(
                kernel_size=kernel_size,
                output_channels=channels,
                stride=1,
                relu_activation=False,
                name="Layer{}".format(n_left),
                **kwargs
            ),
        ]

        layers = [
            [
                _LayerPipe(left_branch_layers, "LeftBranch"),
                _LayerPipe(right_branch_layers, "RightBranch"),
            ],
            kr.layers.Add(name="Sum"),
            kr.layers.ReLU(name="ReLU"),
        ]

        # initialization
        super(_GatherExpansion, self).__init__(layers, "GatherExpansionBlock")


class _ContextEmbedding(_LayerPipe):
    """
    generate the Context Embedding layer(s) of the Semantic branch being part of the BiSeNet V2.

    Parameters
    ----------
    kernel_size: int
        the basic size of the convolutional kernel for each level of the block.

    channels: int
        the basic number of channels of each block.

    kwargs: any
        additional parameters passed to the convolutional layers.
    """

    def __init__(self, kernel_size, channels, **kwargs):

        # check the entries
        assert isinstance(kernel_size, int), "'kernel_size' must be an int object."
        assert isinstance(channels, int), "'channels' must be an int object."

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
                        conv2d_bn_relu(
                            kernel_size=1,
                            output_channels=channels,
                            stride=1,
                            relu_activation=False,
                            name="Layer2",
                            **kwargs
                        ),
                    ],
                    name="LeftBranch",
                ),
                _Identity(name="RightBranch-Layer1-Identity"),
            ],
            kr.layers.Add(name="Layer3-Add"),
            conv2d_bn_relu(
                kernel_size=kernel_size,
                output_channels=channels,
                stride=1,
                relu_activation=False,
                batch_normalization=False,
                name="Layer4",
                **kwargs
            ),
        ]

        # init
        super(_ContextEmbedding, self).__init__(layers, "ContextEmbedding")


class _SemanticBranch(_LayerPipe):
    """
    generate the Semantic Branch of the BiSeNet V2 by concatenating a set of convolutional blocks.

    Parameters
    ----------
    kernel_size: list or tuple, optional
        the size of the convolutional kernel for each level of the block.

    channels: int
        the basic number of channels of each block.

    kwargs: any
        additional parameters passed to the convolutional layers.
    """

    def __init__(self, kernel_size, channels, **kwargs):

        # check the entries
        txt = lambda x, c: "{} must be and object of class {}.".format(x, c)
        assert isinstance(kernel_size, int), txt("kernel_size", int)
        assert isinstance(channels, int), txt("channels", int)

        # create the blocks
        layers = [
            _LayerPipe(
                layers=[_Stem(kernel_size, int(channels / 4), **kwargs)],
                name="Stages1-2",
            ),
            _LayerPipe(
                layers=[
                    _GatherExpansion(kernel_size, int(channels / 2), 2),
                    _GatherExpansion(kernel_size, int(channels / 2), 1),
                ],
                name="Stage3",
            ),
            _LayerPipe(
                layers=[
                    _GatherExpansion(kernel_size, channels, 2),
                    _GatherExpansion(kernel_size, channels, 1),
                ],
                name="Stage4",
            ),
            _LayerPipe(
                layers=[
                    _GatherExpansion(kernel_size, channels * 2, 2),
                    _GatherExpansion(kernel_size, channels * 2, 1),
                    _ContextEmbedding(kernel_size, channels * 2),
                ],
                name="Stage5",
            ),
        ]

        # initialize
        super(_SemanticBranch, self).__init__(layers, "SemanticBranch")


class _BilateralGuidedAggregation(_LayerPipe):
    """
    generate the Aggregation Branch allowing to join together the Detail and Semantic branches from which
    the BiSeNet V2 started.

    Parameters
    ----------
    kernel_size: list or tuple, optional
        the size of the convolutional kernel for each level of the block.

    channels: int
        the basic number of channels of each block.
    """

    def __init__(self, kernel_size, channels):

        # check the entries
        assert isinstance(kernel_size, int), "'kernel_size' must be an int object."
        assert isinstance(channels, int), "'channels' must be an int object."

        # make the branches
        layers = [
            [
                _LayerPipe(
                    name="LeftBranch",
                    share_inputs=False,
                    layers=[
                        [  # Detail part
                            depthwise_conv2_bn(
                                kernel_size=kernel_size,
                                stride=1,
                                multiplier=1,
                                name="Layer1",
                            ),
                            conv2d(
                                kernel_size=1,
                                stride=1,
                                output_channels=channels,
                                name="Layer2",
                            ),
                        ],
                        [  # Semantic part
                            conv2d_bn_relu(
                                kernel_size=kernel_size,
                                stride=1,
                                output_channels=channels,
                                relu_activation=False,
                                name="Layer1",
                            ),
                            _LayerPipe(
                                layers=[
                                    kr.layers.UpSampling2D(
                                        size=4,
                                        interpolation="bilinear",
                                        name="4x Upsampling",
                                    ),
                                    kr.layers.Activation(
                                        kr.activations.sigmoid, name="Sigmoid"
                                    ),
                                ],
                                name="Layer2",
                            ),
                        ],
                        kr.layers.Multiply(name="Layer3-LeftBranch-Multiply"),
                    ],
                ),
                _LayerPipe(
                    name="RightBranch",
                    share_inputs=False,
                    layers=[
                        [  # detail part
                            conv2d_bn_relu(
                                kernel_size=kernel_size,
                                stride=2,
                                output_channels=channels,
                                relu_activation=False,
                                name="Layer1",
                            ),
                            kr.layers.GlobalAvgPool2D(
                                kernel_size=kernel_size, stride=2
                            ),
                        ],
                        [  # semantic part
                            depthwise_conv2_bn(
                                kernel_size=kernel_size,
                                stride=1,
                                multiplier=1,
                                name="Layer1",
                            ),
                            _LayerPipe(
                                layers=[
                                    conv2d(
                                        kernel_size=1,
                                        stride=1,
                                        output_channels=channels,
                                    ),
                                    kr.layers.Activation(kr.activations.sigmoid),
                                ],
                                name="Layer2",
                            ),
                        ],
                        kr.layers.Multiply(name="Layer3-RightBranch-Multiply"),
                    ],
                ),
            ],
            kr.layers.Add(name="Layer4-Sum"),
            conv2d_bn_relu(
                kernel_size=kernel_size,
                stride=1,
                output_channels=channels,
                relu_activation=False,
                name="Layer5",
            ),
        ]

        # initialize
        super(_BilateralGuidedAggregation, self).__init__(layers, "Guided Aggregation")


class _SegmentationHead(_LayerPipe):
    """
    generate the Aggregation Branch allowing to join together the Detail and Semantic branches from which
    the BiSeNet V2 started.

    Parameters
    ----------
    kernel_size: list or tuple, optional
        the size of the convolutional kernel for each level of the block.

    channels: int
        the basic number of channels of each block.

    upsampling_rate: int
        decide how much the output image should be expanded/contracted to fit the output image.

    output_channels: int
        the number of output channels for the image.

    kwargs: any
        Additional parameters passed to the convolutional layers.
    """

    def __init__(self, kernel_size, channels, upsampling_rate, output_channels):

        # check the entries
        assert isinstance(kernel_size, int), "'kernel_size' must be an int object."
        assert isinstance(channels, int), "'channels' must be an int object."
        assert isinstance(
            upsampling_rate, int
        ), "'upsampling_rate' must be an int object."
        assert isinstance(
            output_channels, int
        ), "'output_channels' must be an int object."

        # get the upsampling factor

        # make the branches
        layers = [
            conv2d_bn_relu(
                kernel_size=kernel_size,
                output_channels=channels,
                stride=1,
                name="Layer1",
            ),
            _LayerPipe(
                layers=[
                    conv2d(
                        kernel_size=1,
                        output_channels=output_channels,
                        stride=1,
                    ),
                    kr.layers.UpSampling2D(
                        size=upsampling_rate,
                        interpolation="bilinear",
                        name="{}x Upsampling".format(upsampling_rate),
                    ),
                ],
                name="Layer2",
            ),
        ]

        # initialize
        super(_SegmentationHead, self).__init__(layers, "Segmentation Head")


class BiSeNet2(kr.Model):
    """
    generate a BiSeNet (V2) model.

    Parameters
    ----------

    kernel_size: int
        the size of the convolutional kernel for each level of the block.

    channels: int
        the default number of channels (i.e. convolutional filters).

    semantic_ratio: float
        the amount of channels to be used for the stem block initializing the semantic branch.

        This value must lie in the (0, 1] range (default is 0.25).

    width_factor: int
        determine how much the number of channels should be expanded at each addition of hidden layers.

    depth_factor: int
        determine how many hidden layers should be provided in the detail and segmentation branches.

        Minimum number of layers is 3.

    expansion_factor: int
        this parameter controls the representative ability of the Semantic branch.

    Returns
    -------
    block: Keras.Model
        the BiSeNet2 model.
    """

    def __init__(self, input_shape, kernel_size=3, channels=64):

        # check the entries
        txt = lambda x, c: "{} must be and object of class {}.".format(x, c)
        assert isinstance(kernel_size, int), txt("kernel_size", int)
        assert isinstance(channels, int), txt("channels", int)
        assert isinstance(input_shape, (tuple, list)), txt("input_shape", (tuple, list))
        assert len(input_shape) == 3, "'input_shape' must have len = 3."
        assert all([isinstance(i, int) for i in input_shape]), txt(
            "input_shape elements", int
        )

        # build the model
        x = kr.layers.Input(name="Input", shape=input_shape)
        db = _DetailBranch(kernel_size, channels)(x)
        sb = _SemanticBranch(kernel_size, channels)(x)
        ba = _BilateralGuidedAggregation(kernel_size, channels)((db, sb))
        y = _SegmentationHead(kernel_size, channels, 8, 1)(ba)
        check = 1
