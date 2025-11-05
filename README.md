# 🧠 Real-Time Gesture Recognition FPGA Accelerator

This project demonstrates a complete **hardware/software co-design workflow** to accelerate a Convolutional Neural Network (CNN) for hand gesture recognition on an embedded Xilinx Zynq UltraScale+ MPSoC platform (Avnet UltraZed-EG).

The objective is to **offload inference from the CPU (ARM) to the FPGA (PL)** to achieve low-latency, energy-efficient inference suitable for embedded applications (robotics, smart devices, HMI).

---

## 🚀 Architecture and Workflow

This project doesn't just train a model; it compiles it into hardware. The complete workflow, from Keras to an FPGA bitstream, is as follows:

1.  **Keras (Float) Training:** A compact CNN is first trained with Keras in `float32` to establish an accuracy baseline.
2.  **QAT Training (QKeras):** The model is converted to **QKeras** and retrained (or fine-tuned) using **Quantization-Aware Training (QAT)**. This adapts the network weights to low-precision arithmetic (e.g., `INT8`) that the FPGA can compute efficiently.
3.  **.h5 Export:** The quantized model is saved in the `.h5` format. **HLS4ML** is capable of reading this file and directly interpreting the QKeras layers to infer the hardware data types (e.g., `ap_fixed<8,2>`).
4.  **HLS Synthesis (HLS4ML):** **HLS4ML** is used to convert the `.h5` graph into optimized **HLS C++ code**. It generates a complete project ready for synthesis.
5.  **Hardware Compilation (Vitis HLS):** **Vitis HLS** (called by the HLS4ML build script) synthesizes the C++ into a hardware **IP block** (RTL - Verilog/VHDL) ready to be imported into a logic design.
6.  **System Integration (Vivado):** The hardware IP is imported into **Vivado** and integrated into a Zynq MPSoC **Block Design**. It is connected to the processor (PS) via an **AXI-Lite** interface (for control) and to the DDR memory via **AXI DMA** (for the image pixel stream). A **bitstream** is then generated.
7.  **Deployment (PYNQ):** The final application runs on **PYNQ** (Python on Zynq) on the UltraZed-EG board. The Python code (running on the ARM CPU) handles:
    * Video capture via OpenCV.
    * Image preprocessing (64x64 resize, normalization).
    * Sending the image to the accelerator (PL) via DMA.
    * Receiving the results (logits) from the PL.
    * Post-processing (Softmax on the CPU) and displaying the recognized gesture.

---

## 🚧 Limitations & Future Work

Dataset: This model is trained on SignMNIST. Performance on real-world camera feeds is poor due to the "domain gap".

Roadmap:

1. Fine-tune the model on a real-world dataset (like HaGRID) to improve camera inference.
2. Complete the full PYNQ integration (scripts/5_run_inference_pynq.py).
3. Benchmark power consumption on the UltraZed board.
