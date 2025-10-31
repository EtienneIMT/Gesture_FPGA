#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_fixed.h"
#include "ap_int.h"

#include "nnet_utils/nnet_code_gen.h"
#include "nnet_utils/nnet_helpers.h"
// hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_activation_stream.h"
#include "nnet_utils/nnet_batchnorm.h"
#include "nnet_utils/nnet_batchnorm_stream.h"
#include "nnet_utils/nnet_conv2d.h"
#include "nnet_utils/nnet_conv2d_stream.h"
#include "nnet_utils/nnet_dense.h"
#include "nnet_utils/nnet_dense_compressed.h"
#include "nnet_utils/nnet_dense_stream.h"
#include "nnet_utils/nnet_merge.h"
#include "nnet_utils/nnet_merge_stream.h"
#include "nnet_utils/nnet_padding.h"
#include "nnet_utils/nnet_padding_stream.h"
#include "nnet_utils/nnet_pooling.h"
#include "nnet_utils/nnet_pooling_stream.h"

// hls-fpga-machine-learning insert weights
#include "weights/s79.h"
#include "weights/b79.h"
#include "weights/s72.h"
#include "weights/b72.h"
#include "weights/s73.h"
#include "weights/b73.h"
#include "weights/s75.h"
#include "weights/b75.h"
#include "weights/s77.h"
#include "weights/b77.h"
#include "weights/w96.h"
#include "weights/b96.h"
#include "weights/s93.h"
#include "weights/b93.h"
#include "weights/s82.h"
#include "weights/b82.h"
#include "weights/s83.h"
#include "weights/b83.h"
#include "weights/s85.h"
#include "weights/b85.h"
#include "weights/s86.h"
#include "weights/b86.h"
#include "weights/w94.h"
#include "weights/b94.h"
#include "weights/s90.h"
#include "weights/b90.h"
#include "weights/s88.h"
#include "weights/b88.h"
#include "weights/w95.h"
#include "weights/b95.h"
#include "weights/s91.h"
#include "weights/b91.h"


// hls-fpga-machine-learning insert layer-config
// Quant_0_scale
struct config79 : nnet::batchnorm_config {
    static const unsigned n_in = N_INPUT_1_1*N_INPUT_2_1*N_INPUT_3_1;
    static const unsigned n_filt = -1;
    static const unsigned n_scale_bias = (n_filt == -1) ? n_in : n_filt;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in, reuse_factor);
    static const bool store_weights_in_bram = false;
    typedef model_default_t bias_t;
    typedef model_default_t scale_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// zp2d_Conv2D_Conv_0
struct config97 : nnet::padding2d_config {
    static const unsigned in_height = 64;
    static const unsigned in_width = 64;
    static const unsigned n_chan = 1;
    static const unsigned out_height = 66;
    static const unsigned out_width = 66;
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// Quant_3_rescale
struct config72 : nnet::batchnorm_config {
    static const unsigned n_in = Quant_3_param0_0*Quant_3_param0_1*Quant_3_param0_2*Quant_3_param0_3;
    static const unsigned n_filt = -1;
    static const unsigned n_scale_bias = (n_filt == -1) ? n_in : n_filt;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in, reuse_factor);
    static const bool store_weights_in_bram = false;
    typedef model_default_t bias_t;
    typedef model_default_t scale_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// Quant_4_rescale
struct config73 : nnet::batchnorm_config {
    static const unsigned n_in = Quant_4_param0_0;
    static const unsigned n_filt = -1;
    static const unsigned n_scale_bias = (n_filt == -1) ? n_in : n_filt;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in, reuse_factor);
    static const bool store_weights_in_bram = false;
    typedef model_default_t bias_t;
    typedef model_default_t scale_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// Quant_6_rescale
struct config75 : nnet::batchnorm_config {
    static const unsigned n_in = Quant_6_param0_0;
    static const unsigned n_filt = -1;
    static const unsigned n_scale_bias = (n_filt == -1) ? n_in : n_filt;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in, reuse_factor);
    static const bool store_weights_in_bram = false;
    typedef model_default_t bias_t;
    typedef model_default_t scale_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// Quant_8_rescale
struct config77 : nnet::batchnorm_config {
    static const unsigned n_in = Quant_8_param0_0;
    static const unsigned n_filt = -1;
    static const unsigned n_scale_bias = (n_filt == -1) ? n_in : n_filt;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in, reuse_factor);
    static const bool store_weights_in_bram = false;
    typedef model_default_t bias_t;
    typedef model_default_t scale_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// Conv2D_Conv_0
struct config96_mult : nnet::dense_config {
    static const unsigned n_in = 9;
    static const unsigned n_out = 16;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned n_zeros = 1;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef Conv2D_Conv_0_accum_t accum_t;
    typedef bias96_t bias_t;
    typedef weight96_t weight_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseLatency<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config96 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = 66;
    static const unsigned in_width = 66;
    static const unsigned n_chan = 1;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = 16;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = 64;
    static const unsigned out_width = 64;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 1;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 66;
    static const unsigned min_width = 66;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    static const unsigned n_partitions = 4096;
    static const unsigned n_pixels = out_height * out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::FillConv2DBuffer<data_T, CONFIG_T>;
    typedef Conv2D_Conv_0_accum_t accum_t;
    typedef bias96_t bias_t;
    typedef weight96_t weight_t;
    typedef config96_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_height = nnet::scale_index_regular<K, S, W>;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_width = nnet::scale_index_regular<K, S, W>;
};
const ap_uint<config96::filt_height * config96::filt_width> config96::pixels[] = {0};

// Quant_0_rescale
struct config93 : nnet::batchnorm_config {
    static const unsigned n_in = OUT_HEIGHT_96*OUT_WIDTH_96*N_FILT_96;
    static const unsigned n_filt = -1;
    static const unsigned n_scale_bias = (n_filt == -1) ? n_in : n_filt;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in, reuse_factor);
    static const bool store_weights_in_bram = false;
    typedef model_default_t bias_t;
    typedef model_default_t scale_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// Relu_0
struct ReLU_config56 : nnet::activ_config {
    static const unsigned n_in = 65536;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef Relu_0_table_t table_t;
};

// Quant_9_scale
struct config82 : nnet::batchnorm_config {
    static const unsigned n_in = OUT_HEIGHT_55*OUT_WIDTH_55*N_FILT_55;
    static const unsigned n_filt = -1;
    static const unsigned n_scale_bias = (n_filt == -1) ? n_in : n_filt;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in, reuse_factor);
    static const bool store_weights_in_bram = false;
    typedef model_default_t bias_t;
    typedef model_default_t scale_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// Quant_9_rescale
struct config83 : nnet::batchnorm_config {
    static const unsigned n_in = OUT_HEIGHT_55*OUT_WIDTH_55*N_FILT_55;
    static const unsigned n_filt = -1;
    static const unsigned n_scale_bias = (n_filt == -1) ? n_in : n_filt;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in, reuse_factor);
    static const bool store_weights_in_bram = false;
    typedef model_default_t bias_t;
    typedef model_default_t scale_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// MaxPool_0
struct config58 : nnet::pooling2d_config {
    static const unsigned in_height = 64;
    static const unsigned in_width = 64;
    static const unsigned n_filt = 16;
    static const unsigned stride_height = 2;
    static const unsigned stride_width = 2;
    static const unsigned pool_height = 2;
    static const unsigned pool_width = 2;

    static const unsigned filt_height = pool_height;
    static const unsigned filt_width = pool_width;
    static const unsigned n_chan = n_filt;

    static const unsigned out_height = 32;
    static const unsigned out_width = 32;
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const bool count_pad = false;
    static const nnet::Pool_Op pool_op = nnet::Max;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned reuse_factor = 1;
    typedef MaxPool_0_accum_t accum_t;
};

// Relu_1
struct ReLU_config60 : nnet::activ_config {
    static const unsigned n_in = 32768;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef Relu_1_table_t table_t;
};

// Quant_10_scale
struct config85 : nnet::batchnorm_config {
    static const unsigned n_in = OUT_HEIGHT_59*OUT_WIDTH_59*N_FILT_59;
    static const unsigned n_filt = -1;
    static const unsigned n_scale_bias = (n_filt == -1) ? n_in : n_filt;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in, reuse_factor);
    static const bool store_weights_in_bram = false;
    typedef model_default_t bias_t;
    typedef model_default_t scale_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// Quant_10_rescale
struct config86 : nnet::batchnorm_config {
    static const unsigned n_in = OUT_HEIGHT_59*OUT_WIDTH_59*N_FILT_59;
    static const unsigned n_filt = -1;
    static const unsigned n_scale_bias = (n_filt == -1) ? n_in : n_filt;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in, reuse_factor);
    static const bool store_weights_in_bram = false;
    typedef model_default_t bias_t;
    typedef model_default_t scale_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// MaxPool_1
struct config62 : nnet::pooling2d_config {
    static const unsigned in_height = 32;
    static const unsigned in_width = 32;
    static const unsigned n_filt = 32;
    static const unsigned stride_height = 2;
    static const unsigned stride_width = 2;
    static const unsigned pool_height = 2;
    static const unsigned pool_width = 2;

    static const unsigned filt_height = pool_height;
    static const unsigned filt_width = pool_width;
    static const unsigned n_chan = n_filt;

    static const unsigned out_height = 16;
    static const unsigned out_width = 16;
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const bool count_pad = false;
    static const nnet::Pool_Op pool_op = nnet::Max;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned reuse_factor = 1;
    typedef MaxPool_1_accum_t accum_t;
};

// Dense_MatMul_18_unnamed_post_FIXED
struct config94 : nnet::dense_config {
    static const unsigned n_in = 8192;
    static const unsigned n_out = 128;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 60258;
    static const unsigned n_nonzeros = 988318;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef Dense_MatMul_18_unnamed_post_FIXED_accum_t accum_t;
    typedef model_default_t bias_t;
    typedef weight94_t weight_t;
    typedef layer94_index index_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseLatency<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// Quant_5_rescale
struct config90 : nnet::batchnorm_config {
    static const unsigned n_in = N_LAYER_94;
    static const unsigned n_filt = -1;
    static const unsigned n_scale_bias = (n_filt == -1) ? n_in : n_filt;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in, reuse_factor);
    static const bool store_weights_in_bram = false;
    typedef model_default_t bias_t;
    typedef model_default_t scale_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// Add_19_unnamed_post_FIXED
struct config65 : nnet::merge_config {
    static const unsigned n_elem = N_LAYER_64;
    static const unsigned reuse_factor = 1;
};

// Relu_2
struct ReLU_config66 : nnet::activ_config {
    static const unsigned n_in = 128;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef Relu_2_table_t table_t;
};

// Quant_11_scale
struct config88 : nnet::batchnorm_config {
    static const unsigned n_in = N_LAYER_64;
    static const unsigned n_filt = -1;
    static const unsigned n_scale_bias = (n_filt == -1) ? n_in : n_filt;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in, reuse_factor);
    static const bool store_weights_in_bram = false;
    typedef model_default_t bias_t;
    typedef model_default_t scale_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// Dense_MatMul_22_unnamed_post_FIXED
struct config95 : nnet::dense_config {
    static const unsigned n_in = 128;
    static const unsigned n_out = 5;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 5;
    static const unsigned n_nonzeros = 635;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef Dense_MatMul_22_unnamed_post_FIXED_accum_t accum_t;
    typedef model_default_t bias_t;
    typedef weight95_t weight_t;
    typedef layer95_index index_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseLatency<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// Quant_7_rescale
struct config91 : nnet::batchnorm_config {
    static const unsigned n_in = N_LAYER_95;
    static const unsigned n_filt = -1;
    static const unsigned n_scale_bias = (n_filt == -1) ? n_in : n_filt;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in, reuse_factor);
    static const bool store_weights_in_bram = false;
    typedef model_default_t bias_t;
    typedef model_default_t scale_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// final_output_layer
struct config69 : nnet::merge_config {
    static const unsigned n_elem = N_LAYER_68;
    static const unsigned reuse_factor = 1;
};



#endif
