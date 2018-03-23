import os
import tensorflow as tf

module_path = os.path.realpath(__file__)
module_dir = os.path.dirname(module_path)
lib_path = os.path.join(module_dir, 'roi_pooling.so')
roi_pooling_module = tf.load_op_library(lib_path)


def roi_pooling(inp, rois, pool_height=2, pool_width=2, return_indices=False):
    """
    The python wrapper for Roi-Pooling TensorFlow operation.

    :param inp: tf.Tensor, with shape like(batch_size, height, width, n_channels) and tf.float32 data type.
    :param rois: tf.Tensor, with shape like(batch_size, 5) and tf.int32 data type,
    giving (batch_index, top, left, bottom, right) of one RoI(Region of Interest).
    :param pool_height: int, the output height of Roi-Pooling layer
    :param pool_width: int, the output width of Roi-Pooling layer
    :param return_indices: bool, default to False. Set to true if you wanna know the exact index of the return
    value.
    :returns
        out: tf.Tensor, with shape like(batch_size, pool_height, pool_width, n_channels) and tf.float32 data type.
        indices: tf.Tensor, with same shape as `out` and tf.int32 data type

    """
    out, indices = roi_pooling_module.roi_pooling(inp, rois, pool_height=pool_height, pool_width=pool_width)
    if return_indices:
        return out, indices
    return out

# TODO the gradient
