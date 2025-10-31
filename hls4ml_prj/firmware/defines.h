#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 64
#define N_INPUT_2_1 64
#define N_INPUT_3_1 1
#define N_INPUT_1_1 64
#define N_INPUT_2_1 64
#define N_INPUT_3_1 1
#define OUT_HEIGHT_97 66
#define OUT_WIDTH_97 66
#define N_CHAN_97 1
#define Quant_4_param0_0 32
#define Quant_6_param0_0 128
#define Quant_8_param0_0 5
#define Quant_3_param0_0 32
#define Quant_3_param0_1 3
#define Quant_3_param0_2 3
#define Quant_3_param0_3 16
#define Quant_3_param0_0 32
#define Quant_3_param0_1 3
#define Quant_3_param0_2 3
#define Quant_3_param0_3 16
#define Quant_4_param0_0 32
#define Quant_6_param0_0 128
#define Quant_8_param0_0 5
#define OUT_HEIGHT_96 64
#define OUT_WIDTH_96 64
#define N_FILT_96 16
#define OUT_HEIGHT_55 64
#define OUT_WIDTH_55 64
#define N_FILT_55 16
#define OUT_HEIGHT_55 64
#define OUT_WIDTH_55 64
#define N_FILT_55 16
#define OUT_HEIGHT_55 64
#define OUT_WIDTH_55 64
#define N_FILT_55 16
#define OUT_HEIGHT_55 64
#define OUT_WIDTH_55 64
#define N_FILT_55 16
#define OUT_HEIGHT_58 32
#define OUT_WIDTH_58 32
#define N_FILT_58 16
#define OUT_HEIGHT_59 32
#define OUT_WIDTH_59 32
#define N_FILT_59 32
#define OUT_HEIGHT_59 32
#define OUT_WIDTH_59 32
#define N_FILT_59 32
#define OUT_HEIGHT_59 32
#define OUT_WIDTH_59 32
#define N_FILT_59 32
#define OUT_HEIGHT_59 32
#define OUT_WIDTH_59 32
#define N_FILT_59 32
#define OUT_HEIGHT_62 16
#define OUT_WIDTH_62 16
#define N_FILT_62 32
#define N_SIZE_0_63 8192
#define N_LAYER_94 128
#define N_LAYER_64 128
#define N_LAYER_64 128
#define N_LAYER_64 128
#define N_LAYER_64 128
#define N_LAYER_95 5
#define N_LAYER_68 5
#define N_LAYER_68 5


// hls-fpga-machine-learning insert layer-precision
typedef nnet::array<ap_fixed<16,6>, 1*1> input_t;
typedef nnet::array<ap_fixed<8,8,AP_RND_CONV,AP_SAT,0>, 1*1> layer79_t;
typedef ap_fixed<16,6> model_default_t;
typedef nnet::array<ap_fixed<8,8,AP_RND_CONV,AP_SAT,0>, 1*1> layer97_t;
typedef nnet::array<ap_fixed<8,8,AP_RND_CONV,AP_SAT,0>, 32*1> layer3_t;
typedef nnet::array<ap_fixed<8,8,AP_RND_CONV,AP_SAT,0>, 128*1> layer4_t;
typedef nnet::array<ap_fixed<8,8,AP_RND_CONV,AP_SAT,0>, 5*1> layer5_t;
typedef nnet::array<ap_fixed<8,8,AP_RND_CONV,AP_SAT_SYM,0>, 16*1> layer44_t;
typedef nnet::array<ap_fixed<25,15>, 16*1> Quant_3_rescale_result_t;
typedef nnet::array<ap_fixed<25,15>, 32*1> Quant_4_rescale_result_t;
typedef nnet::array<ap_fixed<25,15>, 128*1> Quant_6_rescale_result_t;
typedef nnet::array<ap_fixed<25,15>, 5*1> Quant_8_rescale_result_t;
typedef ap_fixed<21,21> Conv2D_Conv_0_accum_t;
typedef nnet::array<ap_fixed<21,21>, 16*1> Conv2D_Conv_0_result_t;
typedef ap_fixed<8,8,AP_RND_CONV,AP_SAT_SYM,0> weight96_t;
typedef ap_fixed<8,8,AP_RND_CONV,AP_SAT,0> bias96_t;
typedef nnet::array<ap_fixed<38,28>, 16*1> Quant_0_rescale_result_t;
typedef nnet::array<ap_fixed<16,6>, 16*1> layer56_t;
typedef ap_fixed<18,8> Relu_0_table_t;
typedef nnet::array<ap_ufixed<8,8,AP_RND_CONV,AP_SAT,0>, 16*1> layer82_t;
typedef nnet::array<ap_fixed<25,15>, 16*1> Quant_9_rescale_result_t;
typedef ap_fixed<25,15> MaxPool_0_accum_t;
typedef nnet::array<ap_fixed<25,15>, 16*1> layer58_t;
typedef ap_fixed<16,6> conv_1_accum_t;
typedef nnet::array<ap_fixed<16,6>, 32*1> layer59_t;
typedef nnet::array<ap_fixed<16,6>, 32*1> layer60_t;
typedef ap_fixed<18,8> Relu_1_table_t;
typedef nnet::array<ap_ufixed<8,8,AP_RND_CONV,AP_SAT,0>, 32*1> layer85_t;
typedef nnet::array<ap_fixed<25,15>, 32*1> Quant_10_rescale_result_t;
typedef ap_fixed<25,15> MaxPool_1_accum_t;
typedef nnet::array<ap_fixed<25,15>, 32*1> layer62_t;
typedef ap_fixed<47,37> Dense_MatMul_18_unnamed_post_FIXED_accum_t;
typedef nnet::array<ap_fixed<47,37>, 128*1> Dense_MatMul_18_unnamed_post_FIXED_result_t;
typedef ap_fixed<8,8,AP_RND_CONV,AP_SAT_SYM,0> weight94_t;
typedef ap_uint<1> layer94_index;
typedef nnet::array<ap_fixed<64,44>, 128*1> Quant_5_rescale_result_t;
typedef nnet::array<ap_fixed<65,45>, 128*1> Add_19_unnamed_post_FIXED_result_t;
typedef nnet::array<ap_fixed<16,6>, 128*1> layer66_t;
typedef ap_fixed<18,8> Relu_2_table_t;
typedef nnet::array<ap_ufixed<8,8,AP_RND_CONV,AP_SAT,0>, 128*1> layer88_t;
typedef ap_fixed<34,24> Dense_MatMul_22_unnamed_post_FIXED_accum_t;
typedef nnet::array<ap_fixed<34,24>, 5*1> Dense_MatMul_22_unnamed_post_FIXED_result_t;
typedef ap_fixed<8,8,AP_RND_CONV,AP_SAT_SYM,0> weight95_t;
typedef ap_uint<1> layer95_index;
typedef nnet::array<ap_fixed<51,31>, 5*1> Quant_7_rescale_result_t;
typedef nnet::array<ap_fixed<52,32>, 5*1> result_t;


#endif
