#include <iostream>

#include "myproject.h"
#include "parameters.h"


void myproject(
    hls::stream<input_t> &global_in,
    hls::stream<result_t> &layer69_out
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS INTERFACE axis port=global_in,layer69_out 
    #pragma HLS DATAFLOW

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<model_default_t, 4096>(s79, "s79.txt");
        nnet::load_weights_from_txt<model_default_t, 4096>(b79, "b79.txt");
        nnet::load_weights_from_txt<model_default_t, 4608>(s72, "s72.txt");
        nnet::load_weights_from_txt<model_default_t, 4608>(b72, "b72.txt");
        nnet::load_weights_from_txt<model_default_t, 32>(s73, "s73.txt");
        nnet::load_weights_from_txt<model_default_t, 32>(b73, "b73.txt");
        nnet::load_weights_from_txt<model_default_t, 128>(s75, "s75.txt");
        nnet::load_weights_from_txt<model_default_t, 128>(b75, "b75.txt");
        nnet::load_weights_from_txt<model_default_t, 5>(s77, "s77.txt");
        nnet::load_weights_from_txt<model_default_t, 5>(b77, "b77.txt");
        nnet::load_weights_from_txt<weight96_t, 144>(w96, "w96.txt");
        nnet::load_weights_from_txt<bias96_t, 16>(b96, "b96.txt");
        nnet::load_weights_from_txt<model_default_t, 65536>(s93, "s93.txt");
        nnet::load_weights_from_txt<model_default_t, 65536>(b93, "b93.txt");
        nnet::load_weights_from_txt<model_default_t, 65536>(s82, "s82.txt");
        nnet::load_weights_from_txt<model_default_t, 65536>(b82, "b82.txt");
        nnet::load_weights_from_txt<model_default_t, 65536>(s83, "s83.txt");
        nnet::load_weights_from_txt<model_default_t, 65536>(b83, "b83.txt");
        nnet::load_weights_from_txt<model_default_t, 32768>(s85, "s85.txt");
        nnet::load_weights_from_txt<model_default_t, 32768>(b85, "b85.txt");
        nnet::load_weights_from_txt<model_default_t, 32768>(s86, "s86.txt");
        nnet::load_weights_from_txt<model_default_t, 32768>(b86, "b86.txt");
        nnet::load_weights_from_txt<weight94_t, 1048576>(w94, "w94.txt");
        nnet::load_weights_from_txt<model_default_t, 128>(b94, "b94.txt");
        nnet::load_weights_from_txt<model_default_t, 128>(s90, "s90.txt");
        nnet::load_weights_from_txt<model_default_t, 128>(b90, "b90.txt");
        nnet::load_weights_from_txt<model_default_t, 128>(s88, "s88.txt");
        nnet::load_weights_from_txt<model_default_t, 128>(b88, "b88.txt");
        nnet::load_weights_from_txt<weight95_t, 640>(w95, "w95.txt");
        nnet::load_weights_from_txt<model_default_t, 5>(b95, "b95.txt");
        nnet::load_weights_from_txt<model_default_t, 5>(s91, "s91.txt");
        nnet::load_weights_from_txt<model_default_t, 5>(b91, "b91.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    hls::stream<layer79_t> layer79_out("layer79_out");
    #pragma HLS STREAM variable=layer79_out depth=4096
    nnet::normalize<input_t, layer79_t, config79>(global_in, layer79_out, s79, b79); // Quant_0_scale

    hls::stream<layer97_t> layer97_out("layer97_out");
    #pragma HLS STREAM variable=layer97_out depth=4356
    nnet::zeropad2d_cl<layer79_t, layer97_t, config97>(layer79_out, layer97_out); // zp2d_Conv2D_Conv_0

    hls::stream<layer3_t> Quant_4_param0("Quant_4_param0");
    #pragma HLS STREAM variable=Quant_4_param0 depth=1
    hls::stream<layer4_t> Quant_6_param0("Quant_6_param0");
    #pragma HLS STREAM variable=Quant_6_param0 depth=1
    hls::stream<layer5_t> Quant_8_param0("Quant_8_param0");
    #pragma HLS STREAM variable=Quant_8_param0 depth=1
    hls::stream<layer44_t> Quant_3_param0("Quant_3_param0");
    #pragma HLS STREAM variable=Quant_3_param0 depth=288
    hls::stream<Quant_3_rescale_result_t> layer72_out("layer72_out");
    #pragma HLS STREAM variable=layer72_out depth=288
    nnet::normalize<layer44_t, Quant_3_rescale_result_t, config72>(Quant_3_param0, layer72_out, s72, b72); // Quant_3_rescale

    hls::stream<Quant_4_rescale_result_t> layer73_out("layer73_out");
    #pragma HLS STREAM variable=layer73_out depth=1
    nnet::normalize<layer3_t, Quant_4_rescale_result_t, config73>(Quant_4_param0, layer73_out, s73, b73); // Quant_4_rescale

    hls::stream<Quant_6_rescale_result_t> layer75_out("layer75_out");
    #pragma HLS STREAM variable=layer75_out depth=1
    nnet::normalize<layer4_t, Quant_6_rescale_result_t, config75>(Quant_6_param0, layer75_out, s75, b75); // Quant_6_rescale

    hls::stream<Quant_8_rescale_result_t> layer77_out("layer77_out");
    #pragma HLS STREAM variable=layer77_out depth=1
    nnet::normalize<layer5_t, Quant_8_rescale_result_t, config77>(Quant_8_param0, layer77_out, s77, b77); // Quant_8_rescale

    hls::stream<Conv2D_Conv_0_result_t> layer96_out("layer96_out");
    #pragma HLS STREAM variable=layer96_out depth=4096
    nnet::conv_2d_cl<layer97_t, Conv2D_Conv_0_result_t, config96>(layer97_out, layer96_out, w96, b96); // Conv2D_Conv_0

    hls::stream<Quant_0_rescale_result_t> layer93_out("layer93_out");
    #pragma HLS STREAM variable=layer93_out depth=4096
    nnet::normalize<Conv2D_Conv_0_result_t, Quant_0_rescale_result_t, config93>(layer96_out, layer93_out, s93, b93); // Quant_0_rescale

    hls::stream<layer56_t> layer56_out("layer56_out");
    #pragma HLS STREAM variable=layer56_out depth=4096
    nnet::relu<Quant_0_rescale_result_t, layer56_t, ReLU_config56>(layer93_out, layer56_out); // Relu_0

    hls::stream<layer82_t> layer82_out("layer82_out");
    #pragma HLS STREAM variable=layer82_out depth=4096
    nnet::normalize<layer56_t, layer82_t, config82>(layer56_out, layer82_out, s82, b82); // Quant_9_scale

    hls::stream<Quant_9_rescale_result_t> layer83_out("layer83_out");
    #pragma HLS STREAM variable=layer83_out depth=4096
    nnet::normalize<layer82_t, Quant_9_rescale_result_t, config83>(layer82_out, layer83_out, s83, b83); // Quant_9_rescale

    hls::stream<layer58_t> layer58_out("layer58_out");
    #pragma HLS STREAM variable=layer58_out depth=1024
    nnet::pooling2d_cl<Quant_9_rescale_result_t, layer58_t, config58>(layer83_out, layer58_out); // MaxPool_0

    hls::stream<layer59_t> layer59_out("layer59_out");
    #pragma HLS STREAM variable=layer59_out depth=1024
    hls::stream<layer60_t> layer60_out("layer60_out");
    #pragma HLS STREAM variable=layer60_out depth=1024
    nnet::relu<layer59_t, layer60_t, ReLU_config60>(layer59_out, layer60_out); // Relu_1

    hls::stream<layer85_t> layer85_out("layer85_out");
    #pragma HLS STREAM variable=layer85_out depth=1024
    nnet::normalize<layer60_t, layer85_t, config85>(layer60_out, layer85_out, s85, b85); // Quant_10_scale

    hls::stream<Quant_10_rescale_result_t> layer86_out("layer86_out");
    #pragma HLS STREAM variable=layer86_out depth=1024
    nnet::normalize<layer85_t, Quant_10_rescale_result_t, config86>(layer85_out, layer86_out, s86, b86); // Quant_10_rescale

    hls::stream<layer62_t> layer62_out("layer62_out");
    #pragma HLS STREAM variable=layer62_out depth=256
    nnet::pooling2d_cl<Quant_10_rescale_result_t, layer62_t, config62>(layer86_out, layer62_out); // MaxPool_1

    auto& layer63_out = layer62_out;
    hls::stream<Dense_MatMul_18_unnamed_post_FIXED_result_t> layer94_out("layer94_out");
    #pragma HLS STREAM variable=layer94_out depth=1
    nnet::dense<layer62_t, Dense_MatMul_18_unnamed_post_FIXED_result_t, config94>(layer63_out, layer94_out, w94, b94); // Dense_MatMul_18_unnamed_post_FIXED

    hls::stream<Quant_5_rescale_result_t> layer90_out("layer90_out");
    #pragma HLS STREAM variable=layer90_out depth=1
    nnet::normalize<Dense_MatMul_18_unnamed_post_FIXED_result_t, Quant_5_rescale_result_t, config90>(layer94_out, layer90_out, s90, b90); // Quant_5_rescale

    hls::stream<Add_19_unnamed_post_FIXED_result_t> layer65_out("layer65_out");
    #pragma HLS STREAM variable=layer65_out depth=1
    nnet::add<Quant_5_rescale_result_t, Quant_6_rescale_result_t, Add_19_unnamed_post_FIXED_result_t, config65>(layer90_out, layer75_out, layer65_out); // Add_19_unnamed_post_FIXED

    hls::stream<layer66_t> layer66_out("layer66_out");
    #pragma HLS STREAM variable=layer66_out depth=1
    nnet::relu<Add_19_unnamed_post_FIXED_result_t, layer66_t, ReLU_config66>(layer65_out, layer66_out); // Relu_2

    hls::stream<layer88_t> layer88_out("layer88_out");
    #pragma HLS STREAM variable=layer88_out depth=1
    nnet::normalize<layer66_t, layer88_t, config88>(layer66_out, layer88_out, s88, b88); // Quant_11_scale

    hls::stream<Dense_MatMul_22_unnamed_post_FIXED_result_t> layer95_out("layer95_out");
    #pragma HLS STREAM variable=layer95_out depth=1
    nnet::dense<layer88_t, Dense_MatMul_22_unnamed_post_FIXED_result_t, config95>(layer88_out, layer95_out, w95, b95); // Dense_MatMul_22_unnamed_post_FIXED

    hls::stream<Quant_7_rescale_result_t> layer91_out("layer91_out");
    #pragma HLS STREAM variable=layer91_out depth=1
    nnet::normalize<Dense_MatMul_22_unnamed_post_FIXED_result_t, Quant_7_rescale_result_t, config91>(layer95_out, layer91_out, s91, b91); // Quant_7_rescale

    nnet::add<Quant_7_rescale_result_t, Quant_8_rescale_result_t, result_t, config69>(layer91_out, layer77_out, layer69_out); // final_output_layer

}

