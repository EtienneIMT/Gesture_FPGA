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
#define N_INPUT_3_1 3
#define OUT_HEIGHT_20 66
#define OUT_WIDTH_20 66
#define N_CHAN_20 3
#define OUT_HEIGHT_2 64
#define OUT_WIDTH_2 64
#define N_FILT_2 8
#define OUT_HEIGHT_2 64
#define OUT_WIDTH_2 64
#define N_FILT_2 8
#define OUT_HEIGHT_5 32
#define OUT_WIDTH_5 32
#define N_FILT_5 8
#define OUT_HEIGHT_21 34
#define OUT_WIDTH_21 34
#define N_CHAN_21 8
#define OUT_HEIGHT_6 32
#define OUT_WIDTH_6 32
#define N_FILT_6 16
#define OUT_HEIGHT_6 32
#define OUT_WIDTH_6 32
#define N_FILT_6 16
#define OUT_HEIGHT_9 16
#define OUT_WIDTH_9 16
#define N_FILT_9 16
#define OUT_HEIGHT_22 18
#define OUT_WIDTH_22 18
#define N_CHAN_22 16
#define OUT_HEIGHT_10 16
#define OUT_WIDTH_10 16
#define N_FILT_10 32
#define OUT_HEIGHT_10 16
#define OUT_WIDTH_10 16
#define N_FILT_10 32
#define OUT_HEIGHT_13 8
#define OUT_WIDTH_13 8
#define N_FILT_13 32
#define N_SIZE_0_14 2048
#define N_LAYER_15 32
#define N_LAYER_15 32
#define N_LAYER_18 5

// hls-fpga-machine-learning insert layer-precision
typedef nnet::array<ap_fixed<8,3>, 3*1> input_t;
typedef nnet::array<ap_fixed<8,3>, 3*1> layer20_t;
typedef ap_fixed<8,3> model_default_t;
typedef nnet::array<ap_fixed<8,3>, 8*1> layer2_t;
typedef ap_fixed<8,1> weight2_t;
typedef ap_fixed<8,1> bias2_t;
typedef nnet::array<ap_fixed<8,3>, 8*1> layer4_t;
typedef ap_fixed<18,8> relu1_table_t;
typedef nnet::array<ap_fixed<8,3>, 8*1> layer5_t;
typedef nnet::array<ap_fixed<8,3>, 8*1> layer21_t;
typedef nnet::array<ap_fixed<8,3>, 16*1> layer6_t;
typedef ap_fixed<8,1> weight6_t;
typedef ap_fixed<8,1> bias6_t;
typedef nnet::array<ap_fixed<8,3>, 16*1> layer8_t;
typedef ap_fixed<18,8> relu2_table_t;
typedef nnet::array<ap_fixed<8,3>, 16*1> layer9_t;
typedef nnet::array<ap_fixed<8,3>, 16*1> layer22_t;
typedef nnet::array<ap_fixed<8,3>, 32*1> layer10_t;
typedef ap_fixed<8,1> weight10_t;
typedef ap_fixed<8,1> bias10_t;
typedef nnet::array<ap_fixed<8,3>, 32*1> layer12_t;
typedef ap_fixed<18,8> relu3_table_t;
typedef nnet::array<ap_fixed<8,3>, 32*1> layer13_t;
typedef nnet::array<ap_fixed<8,3>, 32*1> layer15_t;
typedef ap_fixed<8,1> weight15_t;
typedef ap_fixed<8,1> bias15_t;
typedef ap_uint<1> layer15_index;
typedef nnet::array<ap_fixed<8,3>, 32*1> layer17_t;
typedef ap_fixed<18,8> relu_fc1_table_t;
typedef nnet::array<ap_fixed<8,3>, 5*1> result_t;
typedef ap_fixed<8,1> weight18_t;
typedef ap_fixed<8,1> bias18_t;
typedef ap_uint<1> layer18_index;

#endif
