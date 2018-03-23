import numpy as np
import tensorflow as tf

from roi_pooling_op import roi_pooling


class RoiPoolingTest(tf.test.TestCase):
    def test_forward_pass_1(self):
        # 4x4 feature map with only 1 channel
        input_value = [[
            [[1], [2], [4], [4]],
            [[3], [4], [1], [2]],
            [[6], [2], [1], [7]],
            [[1], [3], [2], [8]]
        ]]
        input_value = np.asarray(input_value, dtype='float32')
        rois_value = [
            [0, 0, 0, 3, 1],
            [0, 2, 2, 3, 3],
            [0, 1, 0, 3, 2]
        ]
        rois_value = np.asarray(rois_value, dtype='int32')
        result = [
            [[[3.], [4.]],
             [[6.], [3.]]],
            [[[1.], [7.]],
             [[2.], [8.]]],
            [[[6.], [4.]],
             [[6.], [3.]]],
        ]
        result = np.asarray(result, dtype='float32')
        # in this case we have 3 RoI pooling operations:
        # * channel 0, rectangular region (0, 0) to (3, 1)
        #              xx..
        #              xx..
        #              xx..
        #              xx..
        #
        # * channel 0, rectangular region (2, 2) to (3, 3)
        #              ....
        #              ....
        #              ..xx
        #              ..xx
        # * channel 0, rectangular region (1, 0) to (3, 2)
        #              ....
        #              xxx.
        #              xxx.
        #              xxx.

        with tf.Session('') as sess:
            out = sess.run(roi_pooling(input_value, rois_value, 2, 2))
            self.assertAllEqual(out, result)


if __name__ == '__main__':
    tf.test.main()
