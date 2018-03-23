#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;
using namespace std;


#define idx4_1(idx, d1, d2, d3, d4) ((idx / d4 / d3 / d2) % d1)
#define idx4_2(idx, d1, d2, d3, d4) ((idx / d4 / d3) % d2)
#define idx4_3(idx, d1, d2, d3, d4) ((idx / d4) % d3)
#define idx4_4(idx, d1, d2, d3, d4) (idx % d4)
#define tuple_to_one(idx1, idx2, idx3, idx4, d1, d2, d3, d4) (idx1*d2*d3*d4+idx2*d3*d4+idx3*d4+idx4)


REGISTER_OP("RoiPooling")
    .Input("input: float32")
    .Input("rois: int32")
    .Attr("pool_height: int")
    .Attr("pool_width: int")
    .Output("output: float32")
    .Output("indices: int32");


int force_within(int x, int left, int right){
    return min(max(x, left), right);
};

void RoiPoolingKernelLauncher(const float* input,
                              const int* rois,
                              int n_rois,
                              int n_channels,
                              int height,
                              int width,
                              int pool_height,
                              int pool_width,
                              float* output,
                              int* indices);

class RoiPoolingOp: public OpKernel{
    private:
        int pool_height;
        int pool_width;
    public:
        explicit RoiPoolingOp(OpKernelConstruction* context): OpKernel(context){
            OP_REQUIRES_OK(context, context->GetAttr("pool_height", &pool_height));
            OP_REQUIRES_OK(context, context->GetAttr("pool_width", &pool_width));
        }

        void Compute(OpKernelContext* context)override{
            const Tensor& input_tensor = context->input(0);
            const Tensor& rois_tensor = context->input(1);

            auto input = input_tensor.flat<float>();
            auto rois = rois_tensor.flat<int32>();

            Tensor* output_tensor = NULL;
            Tensor* indices_tensor = NULL;

            auto input_shape = input_tensor.shape();   // [batch_size, height, width, n_channels]
            auto rois_shape = rois_tensor.shape();     // [batch_size, top, left, bottom, right]

            int n_rois       = rois_shape.dim_size(0);
            int input_height = input_shape.dim_size(1);
            int input_width  = input_shape.dim_size(2);
            int n_channels   = input_shape.dim_size(3);

            TensorShape output_shape = TensorShape({static_cast<int64>(n_rois),
                                                    static_cast<int64>(pool_height),
                                                    static_cast<int64>(pool_width),
                                                    static_cast<int64>(n_channels)});

            OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
            OP_REQUIRES_OK(context, context->allocate_output(1, output_shape, &indices_tensor));

            auto output = output_tensor->flat<float>();
            auto indices = indices_tensor->flat<int32>();

            RoiPoolingKernelLauncher(input.data(),
                                     rois.data(),
                                     n_rois,
                                     n_channels,
                                     input_height,
                                     input_width,
                                     pool_height,
                                     pool_width,
                                     output.data(),
                                     indices.data());
        }
};


void RoiPoolingKernelLauncher(const float* input,
                              const int* rois,
                              int n_rois,
                              int n_channels,
                              int height,
                              int width,
                              int pool_height,
                              int pool_width,
                              float* output,
                              int* indices){
    int N = n_rois * pool_height * pool_width * n_channels;
    for(int i = 0; i < N; i++){
        // (n, h, w, c) indexed into the output tensor
        int n = idx4_1(i, n_rois, pool_height, pool_width, n_channels);
        int h = idx4_2(i, n_rois, pool_height, pool_width, n_channels);
        int w = idx4_3(i, n_rois, pool_height, pool_width, n_channels);
        int c = idx4_4(i, n_rois, pool_height, pool_width, n_channels);

        int roi_batch_idx = rois[n*5];
        int roi_top       = rois[n*5 + 1];
        int roi_left      = rois[n*5 + 2];
        int roi_bottom    = rois[n*5 + 3];
        int roi_right     = rois[n*5 + 4];
        int roi_height = max(1, roi_bottom - roi_top + 1);
        int roi_width  = max(1, roi_right - roi_left + 1);

        float bin_h = static_cast<float>(roi_height) / pool_height;
        float bin_w = static_cast<float>(roi_width) / pool_width;

        int hstart = static_cast<int>(floor(h * bin_h));
        int wstart = static_cast<int>(floor(w * bin_w));
        int hend   = static_cast<int>(ceil((h+1) * bin_h));
        int wend   = static_cast<int>(ceil((w+1) * bin_w));

        // force the index within range
        hstart = force_within(hstart + roi_top, 0, height);
        hend   = force_within(hend + roi_top, 0, height);
        wstart = force_within(wstart + roi_left, 0, width);
        wend   = force_within(wend + roi_left, 0, width);


        bool is_empty = (hend <= hstart) || (wend <= wstart);

        float maxval = is_empty ? 0: -99999999.0;
        int maxidx = -1;

        // loop over the (hstart, wstart, hend, wend) sub-window
        // record the maximum value and the related index
        // the max pooling can be easily replace with other strategy(say average pooling)
        for(int idx_h = hstart; idx_h < hend; idx_h++){
            for(int idx_w = wstart; idx_w < wend; idx_w++){
                int input_idx = tuple_to_one(roi_batch_idx, idx_h, idx_w, c, 0, height, width, n_channels);
                if(input[input_idx] > maxval){
                    maxval = input[input_idx];
                    maxidx = input_idx;
                }
            }
        }
        output[i] = maxval;
        indices[i] = maxidx;
    }
};


REGISTER_KERNEL_BUILDER(Name("RoiPooling").Device(DEVICE_CPU), RoiPoolingOp);