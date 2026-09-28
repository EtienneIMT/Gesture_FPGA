# Real-Time Gesture Recognition FPGA Accelerator

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Vitis HLS](https://img.shields.io/badge/Xilinx-Vitis_HLS-black.svg)](https://www.xilinx.com/products/design-tools/vitis.html)

This repository contains a complete hardware/software co-design workflow to accelerate Convolutional Neural Network (CNN) inference for hand gesture recognition. The architecture offloads compute-intensive inference from the CPU to the Programmable Logic (PL) of an FPGA. 

*Note: The project was initially targeted and optimized for a Xilinx Zynq UltraScale+ MPSoC (Avnet UltraZed-EG) using `hls4ml`. The final physical integration and Vivado Block Design are implemented for a Xilinx Zynq-7000 SoC (ZedBoard).*

## Architecture and Workflow

The project establishes an automated pipeline from a high-level TensorFlow/Keras model to a synthesized hardware IP block:

1. **Baseline Training:** A compact CNN is trained on the SignMNIST dataset using Keras (float32). Data preprocessing isolates the hand using MediaPipe.
2. **Quantization-Aware Training (QAT):** The model is converted to QKeras and fine-tuned for low-precision arithmetic (e.g., 6-bit INT) to map efficiently to DSP slices and logic fabric on the FPGA.
3. **Hardware IP Generation:** The quantized model is parsed by `hls4ml`, which infers hardware data types and generates optimized C++ code for High-Level Synthesis.
4. **HLS Synthesis:** Vitis HLS synthesizes the C++ design into a streaming RTL IP block (Verilog/VHDL) using AXI-Stream interfaces.
5. **System Integration:** The generated IP is integrated into a Vivado Block Design. It connects to the Processing System (PS) via AXI-Lite for control registers and utilizes AXI DMA for high-throughput pixel streaming from DDR memory.

## Project Structure

```text
Gesture_FPGA/
├── src/                    # Core libraries and settings
│   ├── settings.py         # Global paths and constants
│   ├── data_loader.py      # Data loading and caching
│   └── infer_test.py       # Inference testing scripts
├── scripts/                # Execution pipeline scripts
│   ├── preprocess_dataset.py # CSV to Image extraction
│   ├── clean_dataset.py    # Hand bounding box extraction (MediaPipe)
│   ├── 1_train_float.py    # Float32 Model Training
│   ├── 2_train_quantized.py # Quantization-Aware Training (QKeras)
│   ├── 3_convert_hls.py    # HLS4ML conversion & Vitis HLS Synthesis
│   └── cam_test.py         # Quick webcam test
├── hw_export/              # Generated hardware handoff files (.bin, .bit, .hwh)
├── zedboard_gesture_system/# Vivado Project for ZedBoard integration
├── IP/                     # Generated IP blocks for Vivado integration
└── README.md
```

## Current Status

The hardware design, synthesis, and Vivado block integration are complete. The hardware handoff files (`.bit`, `.hwh`) are available in the `hw_export/` directory. The project is currently pending physical board bring-up and software driver integration (e.g., PYNQ overlay).

## Getting Started

### Prerequisites
- Python 3.8+
- TensorFlow 2.x
- QKeras
- hls4ml
- Xilinx Vivado & Vitis HLS (Tested with 2020.2 / 2022.2)

### Running the Pipeline
To re-run the end-to-end workflow:

1. **Prepare the Data:** 
   Place your raw SignMNIST CSV files in `data/raw/` and run the preprocessing:
   ```bash
   python scripts/preprocess_dataset.py
   python scripts/clean_dataset.py
   ```
2. **Train the Baseline Model:**
   ```bash
   python scripts/1_train_float.py
   ```
3. **Train the Quantized Model (QAT):**
   ```bash
   python scripts/2_train_quantized.py
   ```
4. **Synthesize to Hardware (HLS):**
   *Note: Ensure your Xilinx tools are sourced in your terminal.*
   ```bash
   python scripts/3_convert_hls.py
   ```
