from __future__ import absolute_import

import os

import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.framework.interpret_util as interpret_util
import oneflow.python.framework.distribute as distribute_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.remote_blob as remote_blob_util
from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("layers.dense")
def dense(
    inputs,
    units,
    activation=None,
    use_bias=True,
    kernel_initializer=None,
    bias_initializer=None,
    kernel_regularizer=None,
    bias_regularizer=None,
    trainable=True,
    name=None,
    model_distribute=distribute_util.broadcast(),
):
    r"""
    dense layer or fully-connected layer

    Args:
        inputs: A 2D input `Blob`.
        units: A positive integer for the dimensionality of the output space.
        activation: Activation function (default: None).
        use_bias: A boolean specifies whether to use a bias vector (default: True).
        kernel_intializer: Initializer for the kernel weights matrix (default: None).
        bias_initializer: Initializer for the bias vector (default: None).
        kernel_regularizer: Regularizer for the kernel weights matrix (default: None).
        bias_regularizer: Regularizer for the bias vector (default: None).
        trainable: A boolean specifies whether to train the variables (default: True).
        name: This layer's name (default: None).
        model_distribute: Define the way to ditribute the model (default: distribute_util.broadcast()).

    Returns:
        A N-D `Blob` with the shape of (batch_size, units).  

    Raises:
        ValueError: The dimension of input `Blob` must be less than 2.
        VauleError: Model distribute must be in auto, broadcast, split.
        ValueError: The input must be a 2D `Blob` when the model distribute is split.
    """
    in_shape = inputs.shape
    in_num_axes = len(in_shape)
    assert in_num_axes >= 2

    name_prefix = name if name is not None else id_util.UniqueStr("Dense_")
    inputs = flow.reshape(inputs, (-1, in_shape[-1])) if in_num_axes > 2 else inputs

    assert (
        model_distribute is distribute_util.auto()
        or model_distribute is distribute_util.broadcast()
        or model_distribute is distribute_util.split(0)
    )

    if model_distribute is distribute_util.split(0):
        assert in_num_axes == 2  # model distribute is hard for reshape split dim 1

    weight = flow.get_variable(
        name="{}-weight".format(name_prefix),
        shape=(units, inputs.shape[1]),
        dtype=inputs.dtype,
        initializer=(
            kernel_initializer
            if kernel_initializer is not None
            else flow.constant_initializer(0)
        ),
        regularizer=kernel_regularizer,
        trainable=trainable,
        model_name="weight",
        distribute=model_distribute,
    )
    weight = weight.with_distribute(model_distribute)

    out = flow.matmul(
        a=inputs, b=weight, transpose_b=True, name="{}_matmul".format(name_prefix),
    )
    if use_bias:
        bias = flow.get_variable(
            name="{}-bias".format(name_prefix),
            shape=(units,),
            dtype=inputs.dtype,
            initializer=(
                bias_initializer
                if bias_initializer is not None
                else flow.constant_initializer(0)
            ),
            regularizer=bias_regularizer,
            trainable=trainable,
            model_name="bias",
            distribute=model_distribute,
        )
        bias = bias.with_distribute(model_distribute)
        out = flow.nn.bias_add(out, bias, name="{}_bias_add".format(name_prefix))
    out = (
        activation(out, name="{}_activation".format(name_prefix))
        if activation is not None
        else out
    )
    out = flow.reshape(out, in_shape[:-1] + (units,)) if in_num_axes > 2 else out

    return out


@oneflow_export("layers.conv2d")
def conv2d(
    inputs,
    filters,
    kernel_size=1,
    strides=1,
    padding="VALID",
    data_format="NCHW",
    dilation_rate=1,
    groups=1,
    activation=None,
    use_bias=True,
    kernel_initializer=None,
    bias_initializer=None,
    kernel_regularizer=None,
    bias_regularizer=None,
    trainable=True,
    name=None,
    weight_name=None,
    bias_name=None,
):
    r"""
    2D convolution layer.

    Args:
        inputs: A 4D input `Blob`.
        filters: An integer specifies the dimensionality of the output space. 
        kernel_size: An integer or tuple/list specifies the height and width of the convolution window. When it is an integer, a square window is applied to the input (default: 1).
        strides:  An integer or tuple/list specifies the strides of the convolution window along the height and width. When it is an integer, the same value for the all spatial dimesions is applied (default: 1).
        padding: "VALID" or "SAME" (default: "VALID").?????
        data_format: A string specifies the format of the input `Blob`, one of "NCHW" or "NHWC" (default: "NCHW"). "NCHW" cooresponds to channels_first, i.e. the input `Blob` with shape (batch_size, channels, height, width). \ 
            "NHWC" cooresponds to channels_last, i.e. the input `Blob` with shape (batch_size, height, width, channels).
        dilation_rate: An integer or tuple/list specifies the dilation rate for the dilated convolution. When it is an integer, the same dilation rate is applied for the all dimensions (default: 1).
        groups: A positive integer specifies number of groups for the Group conv (default: 1).
        activation: Activation function (default: None).
        use_bias: A boolean specifies whether to use a bias vector (default: True).
        kernel_initializer: Initializer for the kernel weights matrix (default: None).
        bias_initializer: Initializer for the bias vector (default: None).
        kernel_regularizer: Regularizer for the kernel weights matrix (default: None).
        bias_regularizer: Regularizer for the bias vector (default: None).
        trainable: A boolean specifies whether to train variables (default: True).
        name: This layer's name (default: None).
        weight_name: This weight's name (default: None).
        bias_name: This bias's name (default: None).

    Returns:
        A 4D `Blob` with the shape of (batch_size, filters, new_height, new_width).  

    Raises:
        ValueError: If the type of kernel_size is not one of integer, list, tuple. 
        ValueError: The number of groups must be positive and number of filters must be divisible by it. 
        ValueError: If data_format is not one of 'NCHW', 'NHWC'. 
        ValueError: If number of input channels is not divisible by number of groups or less than number of groups.
        ValueError: Number of group must be one when data_format is 'NHWC'.
    """

    name_prefix = name if name is not None else id_util.UniqueStr("Conv2D_")
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    else:
        assert isinstance(kernel_size, (list, tuple))
        kernel_size = tuple(kernel_size)

    assert isinstance(groups, int)
    assert groups > 0
    assert groups <= filters
    assert filters % groups == 0
    if data_format.upper() == "NCHW":
        assert groups <= inputs.shape[1]
        assert inputs.shape[1] % groups == 0
        weight_shape = (filters, inputs.shape[1] // groups) + kernel_size
    elif data_format.upper() == "NHWC":
        assert groups == 1
        assert groups <= inputs.shape[3]
        assert inputs.shape[3] % groups == 0
        weight_shape = (
            filters,
            kernel_size[0],
            kernel_size[1],
            inputs.shape[3] // groups,
        )
    else:
        raise ValueError("data_format must be in NCHW or NHWC")

    weight = flow.get_variable(
        weight_name if weight_name else name_prefix + "-weight",
        shape=weight_shape,
        dtype=inputs.dtype,
        initializer=kernel_initializer
        if kernel_initializer is not None
        else flow.constant_initializer(0),
        regularizer=kernel_regularizer,
        trainable=trainable,
        model_name="weight",
    )

    output = flow.nn.conv2d(
        inputs,
        weight,
        strides,
        padding,
        data_format,
        dilation_rate,
        groups=groups,
        name=name,
    )
    if use_bias:
        bias = flow.get_variable(
            bias_name if bias_name else name_prefix + "-bias",
            shape=(filters,),
            dtype=inputs.dtype,
            initializer=bias_initializer
            if bias_initializer is not None
            else flow.constant_initializer(0),
            regularizer=bias_regularizer,
            trainable=trainable,
            model_name="bias",
        )
        output = flow.nn.bias_add(
            output, bias, data_format, name=name_prefix + "-bias_add"
        )
    if activation is not None:
        output = activation(output, name=name_prefix + "-activation")

    return output


@oneflow_export("layers.layer_norm")
def layer_norm(
    inputs,
    center=True,
    scale=True,
    trainable=True,
    begin_norm_axis=1,
    begin_params_axis=-1,
    epsilon=1e-5,
    name=None,
):
    r"""
    layer normalization

    Args:
        inputs: Input `Blob`.
        center: A boolean specifies whether to shift input `Blob` (default: True).
        scale: A boolean specifies whether to scaleinput `Blob` (default: True).
        trainable: A boolean specifies whether to train variables (default: True).
        begin_norm_axis: An integer specifies which axis to normalize at first (default: 1).
        begin_params_axis: An integer (default: -1).
        epsilon: A small float is added to avoid division by zero (default: 1e-5).
        name: This layer's name (default: None).

    Returns:
        A normalized `Blob` with same shape of input. 
    """

    name = name if name is not None else id_util.UniqueStr("LayerNorm_")
    op = (
        flow.user_op_builder(name)
        .Op("layer_norm")
        .Input("x", [inputs])
        .Output("y")
        .Output("mean")
        .Output("inv_variance")
    )
    if center == False and scale == False:
        trainable = False
    param_shape = inputs.shape[begin_params_axis:]
    if center:
        beta = flow.get_variable(
            name="{}-beta".format(name),
            shape=param_shape,
            dtype=inputs.dtype,
            initializer=flow.constant_initializer(0.0),
            trainable=trainable,
            model_name="beta",
            distribute=distribute_util.broadcast(),
        )
        op.Input("beta", [beta])
    if scale:
        gamma = flow.get_variable(
            name="{}-gamma".format(name),
            shape=param_shape,
            dtype=inputs.dtype,
            initializer=flow.constant_initializer(1.0),
            trainable=trainable,
            model_name="gamma",
            distribute=distribute_util.broadcast(),
        )
        op.Input("gamma", [gamma])
        op.Output("normalized")
    op.Attr("center", center, "AttrTypeBool")
    op.Attr("scale", scale, "AttrTypeBool")
    op.Attr("begin_norm_axis", begin_norm_axis, "AttrTypeInt64")
    op.Attr("begin_params_axis", begin_params_axis, "AttrTypeInt64")
    op.Attr("epsilon", epsilon, "AttrTypeDouble")
    return op.Build().InferAndTryRun().RemoteBlobList()[0]


@oneflow_export("layers.layer_norm_grad")
def layer_norm_grad(
    dy, x, mean, inv_variance, begin_norm_axis=1, name=None,
):
    r"""
    layer normalization

    Args:
        dy: Upstream derivstives.
        x: Input `Blob`.
        mean: Mean over neurons.
        inv_variance: Variance over neurons.
        begin_norm_axis: An integer specifies which axis to normalize at first (default: 1). 
        name: This layer's name (default: None).

    Returns:
        Gradient with respect to input `Blob`. 
    """

    name = name if name is not None else id_util.UniqueStr("LayerNormGrad_")
    op = (
        flow.user_op_builder(name)
        .Op("layer_norm_grad")
        .Input("dy", [dy])
        .Input("x", [x])
        .Input("mean", [mean])
        .Input("inv_variance", [inv_variance])
        .Output("dx")
        .Attr("begin_norm_axis", begin_norm_axis, "AttrTypeInt64")
        .Attr("epsilon", 1e-5, "AttrTypeDouble")
    )
    return op.Build().InferAndTryRun().RemoteBlobList()[0]


@oneflow_export("layers.layer_norm_param_grad")
def layer_norm_param_grad(
    dy, norm, gamma, begin_params_axis=-1, name=None,
):
    r"""
        backward pass for layer normalization

    Args:
        dy: Upstream derivstives.
        norm: Normalized output. 
        gamma: Scale parameter.  
        begin_params_axis: From which parameters to begin with (default: -1). 
        name: This layer's name (default: None).

    Returns:
        normalized_diff: Gradient with respect to input `Blob`.
        beta_diff: Gradient with respect to shift parameter beta.
        gamma_diff: Gradient with respect to scale parameter gamma.
    """
    name = name if name is not None else id_util.UniqueStr("LayerNormGrad_")
    normalized_diff, beta_diff, gamma_diff, reduce_buf = (
        flow.user_op_builder(name)
        .Op("layer_norm_param_grad")
        .Input("dy", [dy])
        .Input("normalized", [norm])
        .Input("gamma", [gamma])
        .Output("normalized_diff")
        .Output("beta_diff")
        .Output("gamma_diff")
        .Output("reduce_buf")
        .Attr("begin_params_axis", begin_params_axis, "AttrTypeInt64")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()
    )
    return normalized_diff, beta_diff, gamma_diff


@oneflow_export("layers.batch_normalization")
def batch_normalization(
    inputs,
    axis=-1,
    momentum=0.99,
    epsilon=0.001,
    center=True,
    scale=True,
    beta_initializer=None,
    gamma_initializer=None,
    beta_regularizer=None,
    gamma_regularizer=None,
    moving_mean_initializer=None,
    moving_variance_initializer=None,
    trainable=True,
    training=True,
    name=None,
):
    r"""
    batch normalization

    Args:
        inputs: Input `Blob`.
        axis: An int specifies the aixs that should be normalized (default: -1). Default is -1, which normalizes the last axis.
        momentum: A float specifies the momontum for the moving average (default: 0.99).
        epsilon: A small float added to avoid division by zero (default: 0.001).
        center: A boolean specifies whether to add offset to normalized `Blob` (default: True).
        scale: A boolean specifies whether to multiply normalized `Blob` by gamma (default: True).
        beta_initializer: Initializer for beta (default: None).
        gamma_initializer: Initializer for gamma (default: None).
        beta_regularizer: Regularizer for beta (default: None).
        gamma_regularizer: Regularizer for gamma (default: None).
        moving_mean_initializer: Initializer for moving mean (default: None).
        moving_variance_initializer: Initializer for moving variance (default: None).
        trainable: A boolean specifies whether to train variables (default: True).
        training: A boolean specifies whether now is training the model (default: True). 
        name: This layer's name (default: None).

    Returns:
        A `Blob` with same shape of input. 

    Raises:
        ValueError: If axis is out of dimension of input.
    """
    assert axis >= -len(inputs.shape) and axis < len(inputs.shape)
    if axis < 0:
        axis += len(inputs.shape)
    params_shape = [inputs.shape[axis]]
    # Float32 required to avoid precision-loss when using fp16 input/output
    params_dtype = flow.float32 if inputs.dtype == flow.float16 else inputs.dtype

    if not flow.current_global_function_desc().IsTrainable() or not trainable:
        training = False

    if name is None:
        name = id_util.UniqueStr("BatchNorm_")

    if center:
        beta = flow.get_variable(
            name=name + "-beta",
            shape=params_shape,
            dtype=params_dtype,
            initializer=beta_initializer or flow.zeros_initializer(),
            regularizer=beta_regularizer,
            trainable=trainable,
            distribute=distribute_util.broadcast(),
        )
    else:
        beta = flow.constant(0, dtype=params_dtype, shape=params_shape)

    if scale:
        gamma = flow.get_variable(
            name=name + "-gamma",
            shape=params_shape,
            dtype=params_dtype,
            initializer=gamma_initializer or flow.ones_initializer(),
            regularizer=gamma_regularizer,
            trainable=trainable,
            distribute=distribute_util.broadcast(),
        )
    else:
        gamma = flow.constant(1, dtype=params_dtype, shape=params_shape)

    moving_mean = flow.get_variable(
        name=name + "-moving_mean",
        shape=params_shape,
        dtype=params_dtype,
        initializer=moving_mean_initializer or flow.zeros_initializer(),
        trainable=False,
        distribute=distribute_util.broadcast(),
    )

    moving_variance = flow.get_variable(
        name=name + "-moving_variance",
        shape=params_shape,
        dtype=params_dtype,
        initializer=moving_variance_initializer or flow.ones_initializer(),
        trainable=False,
        distribute=distribute_util.broadcast(),
    )

    builder = (
        flow.user_op_builder(name)
        .Op("normalization")
        .Input("x", [inputs])
        .Input("moving_mean", [moving_mean])
        .Input("moving_variance", [moving_variance])
        .Input("gamma", [gamma])
        .Input("beta", [beta])
        .Output("y")
        .Attr("axis", axis, "AttrTypeInt32")
        .Attr("epsilon", epsilon, "AttrTypeFloat")
        .Attr("training", training, "AttrTypeBool")
        .Attr("momentum", momentum, "AttrTypeFloat")
    )
    if trainable and training:
        builder = builder.Output("mean").Output("inv_variance")
    return builder.Build().InferAndTryRun().RemoteBlobList()[0]
