# Real-Time Gesture Recognition FPGA Accelerator

This repository contains a hardware/software co-design workflow to accelerate Convolutional Neural Network (CNN) inference for hand gesture recognition. The architecture targets a Xilinx Zynq UltraScale+ MPSoC (Avnet UltraZed-EG), offloading compute-intensive inference from the ARM CPU to the Programmable Logic (PL).

## Architecture and Workflow

The project establishes an automated pipeline from a high-level TensorFlow/Keras model to a synthesized hardware IP block:

1. **Baseline Training:** A compact CNN is trained on the SignMNIST dataset using Keras (float32).
2. **Quantization-Aware Training (QAT):** The model is converted to QKeras and fine-tuned for low-precision arithmetic (e.g., INT8) to map efficiently to DSP slices and logic fabric.
3. **Hardware IP Generation:** The quantized `.h5` model is parsed by HLS4ML, which infers hardware data types and generates optimized C++ code for High-Level Synthesis.
4. **HLS Synthesis:** Vitis HLS synthesizes the C++ design into an RTL IP block (Verilog/VHDL).
5. **System Integration:** The generated IP is integrated into a Vivado Block Design. It connects to the Processing System (PS) via AXI-Lite for control registers and utilizes AXI DMA for high-throughput pixel streaming from DDR memory.

## Current Status & Roadmap

The hardware design, synthesis, and Vivado block integration are complete. The project is currently pending physical board bring-up and software driver integration. 

Future development phases include:
* **Physical Deployment:** Finalize the PYNQ overlay integration to handle OpenCV video capture, image preprocessing, and AXI DMA transfers on the ARM CPU.
* **Dataset Transition:** Address the current domain gap by fine-tuning the model on real-world camera feeds (e.g., HaGRID dataset) rather than static SignMNIST images.
* **Hardware Benchmarking:** Measure end-to-end inference latency and active power consumption on the UltraZed-EG board.
