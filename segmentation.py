# IMPORTS

import tensorflow.keras as kr
import tensorflow as tf


# METHODS


def bisenet2(input_shape, kernel_size=3, channels=64):
    """
    generate a BiSeNet (V2) model.

    Parameters
    ----------

    kernel_size: int
        the size of the convolutional kernel for each level of the block.

    channels: int
        the default number of channels (i.e. convolutional filters).

    Returns
    -------
    block: Keras.Model
        the BiSeNet2 model.
    """

    def _message(who, what, cls):
        """
        return an error message.
        """
        txt = "{} must be and object of class {}, while if was found to be {}."
        return txt.format(who, what, cls)

    def _is_pow_of(x, p):
        """
        check if x is a power of p.

        Parameters
        ----------
        x: int, float
            the number to test

        p: int, float
            the power to check.

        Returns
        -------
        check: bool
            True if x is a power of p, False otherwise.
        """
        assert isinstance(x, (int, float)), _message("x", (int, float), x.__class__)
        assert isinstance(p, (int, float)), _message("p", (int, float), p.__class__)
        assert x > 0, "'x' must be > 0."
        n = 0
        while p ** n < x:
            n += 1
        return p ** n == x

    # check the entries
    assert isinstance(kernel_size, int), _message(
        "kernel_size", int, kernel_size.__class__.__name__
    )
    assert isinstance(channels, int), _message(
        "channels", int, channels.__class__.__name__
    )
    assert isinstance(input_shape, (tuple, list)), _message(
        "input_shape", (tuple, list), input_shape.__class__.__name__
    )
    assert len(input_shape) == 3, "'input_shape' must have len = 3."
    for i in input_shape:
        assert isinstance(i, int), _message(
            "input_shape elements", int, i.__class__.__name__
        )
    for i in input_shape[:-1]:
        assert _is_pow_of(i, 2), "input shape dimensions must be a power of 2."

    # build the model
    x = kr.layers.Input(input_shape)
    db = DetailBranch(
        kernel_size=kernel_size,
        channels=channels,
    )
    sb = SemanticBranch(
        kernel_size=kernel_size,
        channels=channels,
        input_shape=input_shape[:-1],
    )
    ba = BilateralGuidedAggregation(
        kernel_size=kernel_size,
        channels=channels,
    )
    sh = SegmentationHead(
        kernel_size=kernel_size,
        channels=channels,
        upsampling_rate=8,
        output_channels=input_shape[-1],
    )
    dbx = db(x)
    sbx = sb(x)
    bax = ba(((dbx, sbx), (dbx, sbx)))
    shx = sh(bax)
    y = kr.layers.Softmax()(shx)
    return kr.models.Model(inputs=x, outputs=y)


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

    # return the pipe
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
        self.mul = 0.5

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
        val = tf.multiply(inputs, self.mul)
        return kr.layers.Add(name=self._name)([val, val])

    def get_config(self):
        cfg = super(_Identity, self).get_config()
        cfg.update({"mul": self.mul})
        return cfg

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

    def __call__(self, inputs, training=False):
        """
        handle a call to the class object and return the concatenated layers.

        Parameters
        ----------
        inputs: kr.layers.Layer or list or tuple
            the Keras layer to be used as input.

        training: bool
            is the call due to a training purpose or a prediction purpose?

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
        def recaller(obj, arg, share_args=False, training=False):
            """
            recursive function to allow the call of nested layers.
            """
            if isinstance(obj, (list, tuple)):
                if share_args:
                    return [recaller(e, arg, share_args, training) for e in obj]
                else:
                    return [recaller(e, a, False, training) for e, a in zip(obj, arg)]

            elif isinstance(obj, _LayerPipe):
                x = arg
                for level in obj.layers:
                    x = recaller(level, x, share_args, training)
                return x

            else:
                return obj(arg)

        # iterate over the layers
        for i, layer in enumerate(self.layers):
            if i == 0:
                x = recaller(layer, x, self.share_inputs, training)
            else:
                x = recaller(layer, x, True, training)
        return x


class DetailBranch(_LayerPipe):
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
                    conv2d(kernel_size, channels, 2, name="lvl1", **kwargs),
                    conv2d(kernel_size, channels, 1, name="lvl2", **kwargs),
                ],
                name="Stage1",
            ),
            _LayerPipe(
                layers=[
                    conv2d(kernel_size, channels, 2, name="lvl1", **kwargs),
                    conv2d(kernel_size, channels, 1, name="lvl2", **kwargs),
                    conv2d(kernel_size, channels, 1, name="lvl3", **kwargs),
                ],
                name="Stage2",
            ),
            _LayerPipe(
                layers=[
                    conv2d(kernel_size, channels * 2, 2, name="lvl1", **kwargs),
                    conv2d(kernel_size, channels * 2, 1, name="lvl2", **kwargs),
                    conv2d(kernel_size, channels * 2, 1, name="lvl3", **kwargs),
                ],
                name="Stage3",
            ),
        ]

        # init
        super(DetailBranch, self).__init__(layers, "Detail Branch")


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
        if any([i == "name" for i in kwargs]):
            name = kwargs["name"]
            kwargs.pop("name")
        else:
            name = ""

        # here the layers are put in lists indicating how to
        # process them by the __call__ method.
        left_branch_layers = [
            conv2d_bn_relu(
                kernel_size=kernel_size,
                output_channels=channels,
                stride=1,
                name="Layer1" if name == "" else "{}-Layer1".format(name),
                **kwargs
            ),
            depthwise_conv2_bn(
                kernel_size=kernel_size,
                stride=stride,
                multiplier=6,
                name="Layer2" if name == "" else "{}-Layer2".format(name),
            ),
        ]
        n_left = 2

        if stride == 2:

            left_branch_layers += [
                depthwise_conv2_bn(
                    kernel_size=kernel_size,
                    stride=1,
                    multiplier=6,
                    name="Layer3" if name == "" else "{}-Layer3".format(name),
                ),
            ]
            n_left += 1
            right_branch_layers = [
                depthwise_conv2_bn(
                    kernel_size=kernel_size,
                    stride=stride,
                    multiplier=6,
                    name="Layer1" if name == "" else "{}-Layer1".format(name),
                ),
                conv2d_bn_relu(
                    kernel_size=1,
                    output_channels=channels,
                    stride=1,
                    name="Layer2" if name == "" else "{}-Layer2".format(name),
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
            kr.layers.Add(name="Sum" if name == "" else "{}-Sum".format(name)),
            kr.layers.ReLU(name="ReLU" if name == "" else "{}-ReLU".format(name)),
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

    def __init__(self, kernel_size, channels, input_shape, **kwargs):

        # check the entries
        assert isinstance(kernel_size, int), "'kernel_size' must be an int object."
        assert isinstance(channels, int), "'channels' must be an int object."
        txt = "'input_shape' must be a list or tuple."
        assert isinstance(input_shape, (tuple, list)), txt
        assert len(input_shape) == 2, "'input_shape' must have len=2."
        txt = "'input_shape' elements must be int."
        assert all([isinstance(i, (int)) for i in input_shape]), txt

        # create the blocks
        layers = [
            [
                _Identity(name="RightBranch-Layer1-Identity"),
                _LayerPipe(
                    layers=[
                        _LayerPipe(
                            layers=[
                                kr.layers.Lambda(
                                    lambda t4d: tf.math.reduce_mean(
                                        t4d, axis=(1, 2), keepdims=True
                                    ),
                                    name="GlobalAveragePooling2D",
                                ),
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


class SemanticBranch(_LayerPipe):
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

    def __init__(self, kernel_size, channels, input_shape, **kwargs):

        # check the entries
        txt = lambda x, c: "{} must be and object of class {}.".format(x, c)
        assert isinstance(kernel_size, int), txt("kernel_size", int)
        assert isinstance(channels, int), txt("channels", int)

        # create the blocks
        context_shape = [int(round(i / 32)) for i in input_shape]
        layers = [
            _LayerPipe(
                layers=[_Stem(kernel_size, int(channels / 4), **kwargs)],
                name="Stages1-2",
            ),
            _LayerPipe(
                layers=[
                    _GatherExpansion(kernel_size, int(channels / 2), 2, name="GE1"),
                    _GatherExpansion(kernel_size, int(channels / 2), 1, name="GE2"),
                ],
                name="Stage3",
            ),
            _LayerPipe(
                layers=[
                    _GatherExpansion(kernel_size, channels, 2, name="GE1"),
                    _GatherExpansion(kernel_size, channels, 1, name="GE2"),
                ],
                name="Stage4",
            ),
            _LayerPipe(
                layers=[
                    _GatherExpansion(kernel_size, channels * 2, 2, name="GE1"),
                    _GatherExpansion(kernel_size, channels * 2, 1, name="GE2"),
                    _ContextEmbedding(kernel_size, channels * 2, context_shape),
                ],
                name="Stage5",
            ),
        ]

        # initialize
        super(SemanticBranch, self).__init__(layers, "SemanticBranch")


class BilateralGuidedAggregation(_LayerPipe):
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
                        [
                            _LayerPipe(
                                name="DetailInput",
                                layers=[
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
                            ),
                            _LayerPipe(
                                name="SemanticInput",
                                layers=[
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
                                                name="4xUpsampling",
                                            ),
                                            kr.layers.Activation(
                                                kr.activations.sigmoid, name="Sigmoid"
                                            ),
                                        ],
                                        name="Layer2",
                                    ),
                                ],
                            ),
                        ],
                        kr.layers.Multiply(name="Layer3-LeftBranch-Multiply"),
                    ],
                ),
                _LayerPipe(
                    name="RightBranch",
                    share_inputs=False,
                    layers=[
                        [
                            _LayerPipe(
                                name="DetailInput",
                                layers=[
                                    conv2d_bn_relu(
                                        kernel_size=kernel_size,
                                        stride=2,
                                        output_channels=channels,
                                        relu_activation=False,
                                        name="Layer1",
                                    ),
                                    kr.layers.AveragePooling2D(
                                        pool_size=kernel_size,
                                        strides=2,
                                        padding="same",
                                    ),
                                ],
                            ),
                            _LayerPipe(
                                name="SemanticInput",
                                layers=[
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
                                            kr.layers.Activation(
                                                kr.activations.sigmoid
                                            ),
                                        ],
                                        name="Layer2",
                                    ),
                                ],
                            ),
                        ],
                        kr.layers.Multiply(name="Layer3-RightBranch-Multiply"),
                        kr.layers.UpSampling2D(
                            size=4,
                            interpolation="bilinear",
                            name="4xUpsampling",
                        ),
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
        super(BilateralGuidedAggregation, self).__init__(
            layers, "Guided Aggregation", False
        )


class SegmentationHead(_LayerPipe):
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
        txt = lambda x, y: "'{}' must be an {} object.".format(x, y)
        assert isinstance(kernel_size, int), txt("kernel_size", int)
        assert isinstance(channels, int), txt("channels", int)
        assert isinstance(upsampling_rate, int), txt("upsampling_rate", int)
        assert isinstance(output_channels, int), txt("output_channels", int)

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
                        name="{}xUpsampling".format(upsampling_rate),
                    ),
                ],
                name="Layer2",
            ),
        ]

        # initialize
        super(SegmentationHead, self).__init__(layers, "Segmentation Head")
