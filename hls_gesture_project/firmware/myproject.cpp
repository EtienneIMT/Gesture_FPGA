#include <iostream>

#include "myproject.h"
#include "parameters.h"

void myproject(
    hls::stream<input_t> &input_1,
    hls::stream<result_t> &layer18_out
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS INTERFACE axis port=input_1,layer18_out 
    #pragma HLS DATAFLOW 

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<weight2_t, 216>(w2, "w2.txt");
        nnet::load_weights_from_txt<bias2_t, 8>(b2, "b2.txt");
        nnet::load_weights_from_txt<weight6_t, 1152>(w6, "w6.txt");
        nnet::load_weights_from_txt<bias6_t, 16>(b6, "b6.txt");
        nnet::load_weights_from_txt<weight10_t, 4608>(w10, "w10.txt");
        nnet::load_weights_from_txt<bias10_t, 32>(b10, "b10.txt");
        nnet::load_weights_from_txt<weight15_t, 65536>(w15, "w15.txt");
        nnet::load_weights_from_txt<bias15_t, 32>(b15, "b15.txt");
        nnet::load_weights_from_txt<weight18_t, 160>(w18, "w18.txt");
        nnet::load_weights_from_txt<bias18_t, 5>(b18, "b18.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    hls::stream<layer20_t> layer20_out("layer20_out");
    #pragma HLS STREAM variable=layer20_out depth=4356
    nnet::zeropad2d_cl<input_t, layer20_t, config20>(input_1, layer20_out); // zp2d_conv1

    hls::stream<layer2_t> layer2_out("layer2_out");
    #pragma HLS STREAM variable=layer2_out depth=4096
    nnet::conv_2d_cl<layer20_t, layer2_t, config2>(layer20_out, layer2_out, w2, b2); // conv1

    hls::stream<layer4_t> layer4_out("layer4_out");
    #pragma HLS STREAM variable=layer4_out depth=4096
    nnet::relu<layer2_t, layer4_t, relu_config4>(layer2_out, layer4_out); // relu1

    hls::stream<layer5_t> layer5_out("layer5_out");
    #pragma HLS STREAM variable=layer5_out depth=1024
    nnet::pooling2d_cl<layer4_t, layer5_t, config5>(layer4_out, layer5_out); // pool1

    hls::stream<layer21_t> layer21_out("layer21_out");
    #pragma HLS STREAM variable=layer21_out depth=1156
    nnet::zeropad2d_cl<layer5_t, layer21_t, config21>(layer5_out, layer21_out); // zp2d_conv2

    hls::stream<layer6_t> layer6_out("layer6_out");
    #pragma HLS STREAM variable=layer6_out depth=1024
    nnet::conv_2d_cl<layer21_t, layer6_t, config6>(layer21_out, layer6_out, w6, b6); // conv2

    hls::stream<layer8_t> layer8_out("layer8_out");
    #pragma HLS STREAM variable=layer8_out depth=1024
    nnet::relu<layer6_t, layer8_t, relu_config8>(layer6_out, layer8_out); // relu2

    hls::stream<layer9_t> layer9_out("layer9_out");
    #pragma HLS STREAM variable=layer9_out depth=256
    nnet::pooling2d_cl<layer8_t, layer9_t, config9>(layer8_out, layer9_out); // pool2

    hls::stream<layer22_t> layer22_out("layer22_out");
    #pragma HLS STREAM variable=layer22_out depth=324
    nnet::zeropad2d_cl<layer9_t, layer22_t, config22>(layer9_out, layer22_out); // zp2d_conv3

    hls::stream<layer10_t> layer10_out("layer10_out");
    #pragma HLS STREAM variable=layer10_out depth=256
    nnet::conv_2d_cl<layer22_t, layer10_t, config10>(layer22_out, layer10_out, w10, b10); // conv3

    hls::stream<layer12_t> layer12_out("layer12_out");
    #pragma HLS STREAM variable=layer12_out depth=256
    nnet::relu<layer10_t, layer12_t, relu_config12>(layer10_out, layer12_out); // relu3

    hls::stream<layer13_t> layer13_out("layer13_out");
    #pragma HLS STREAM variable=layer13_out depth=64
    nnet::pooling2d_cl<layer12_t, layer13_t, config13>(layer12_out, layer13_out); // pool3

    auto& layer14_out = layer13_out;
    hls::stream<layer15_t> layer15_out("layer15_out");
    #pragma HLS STREAM variable=layer15_out depth=1
    nnet::dense<layer13_t, layer15_t, config15>(layer14_out, layer15_out, w15, b15); // fc1

    hls::stream<layer17_t> layer17_out("layer17_out");
    #pragma HLS STREAM variable=layer17_out depth=1
    nnet::relu<layer15_t, layer17_t, relu_config17>(layer15_out, layer17_out); // relu_fc1

    nnet::dense<layer17_t, result_t, config18>(layer17_out, layer18_out, w18, b18); // output_logits

}
