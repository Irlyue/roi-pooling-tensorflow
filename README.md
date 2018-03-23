# Roi-Pooling

Best explanation about how Roi-Pooling layer work can be found in this [blog](https://deepsense.io/region-of-interest-pooling-explained/).
Here is a GIF from that blog.

![](https://github.com/deepsense-ai/roi-pooling/blob/master/roi_pooling_animation.gif)

## Q & A

1. Will the sub-window overlap with each other?

   Yes. If the size of the sub-window isn't an integer, the adjacent sub-window will actually overlap with each other. It can be easily spot from the code where we're calculating the position of each individual sub-window:

   ```C++
   // floor for start and ceil for end
   int hstart = static_cast<int>(floor(h * bin_h));
   int wstart = static_cast<int>(floor(w * bin_w));
   int hend   = static_cast<int>(ceil((h+1) * bin_h));
   int wend   = static_cast<int>(ceil((w+1) * bin_w));
   ```

   Consider a 3\*3 RoI and we wanna get a 2\*2 output, so the sub-window size is actually 2\*2.

   ```Python
   [[1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]]
   # sub-windows
   #(0, 0)   (0, 1)
   [[1, 2],  [[2, 3],
    [4, 5]]   [5, 6]]
   #(1, 0)   (1, 1)
   [[4, 5],  [[5, 6],
    [7, 8]]   [8, 9]]

   ```

   ​