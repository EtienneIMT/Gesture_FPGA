"""
Convert the quantized Keras model to a C++ HLS representation and synthesize 
it into a hardware IP block using Vitis HLS.
"""

import sys
import os
import tensorflow as tf
from tensorflow import keras
import qkeras
from qkeras import QConv2D, QDense, QActivation
from qkeras.quantizers import quantized_bits, quantized_relu
import hls4ml
from hls4ml.model.profiling import types_hlsmodel

# Ensure the root directory is on the path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.settings import MODEL_QAT_PATH, HLS_PROJECT_PATH

def create_hls_config(model):
    """Generates and optimizes the hls4ml configuration for the target FPGA."""
    config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    print("Base structure generated. Applying hardware optimizations...")

    # Global Parameters
    config["ProjectName"] = "hls_gesture_model"
    config["OutputDir"] = HLS_PROJECT_PATH
    config["Part"] = "xc7z020clg484-1" # Targeting Zynq-7000 (Zedboard) for synthesis
    config["ClockPeriod"] = 10
    config["IOType"] = "io_stream"

    # Global Strategy
    config["Model"] = {
        "Precision": "ap_fixed<10,4>",
        "ReuseFactor": 512, # Enforce serialization to save resources
        "Strategy": "Resource"
    }

    # Layer-by-layer optimization
    for layer in config['LayerName'].keys():
        config['LayerName'][layer]['ReuseFactor'] = 512
        config['LayerName'][layer]['Strategy'] = 'Resource'
        
        # Optimize convolutions for streaming
        if 'conv' in layer:
            config['LayerName'][layer]['ReuseFactor'] = 512
            config['LayerName'][layer]['ConvImplementation'] = 'LineBuffer' 

    if 'fc1' in config['LayerName']:
        print(f"Configuration for 'fc1': {config['LayerName']['fc1']}")

    return config


if __name__ == "__main__":
    # 1. Re-register QKeras objects for model loading
    custom_objects = {}
    for layer_type in [QConv2D, QDense, QActivation, quantized_bits, quantized_relu]:
        custom_objects[layer_type.__name__] = layer_type

    # 2. Load the quantized model
    print(f"Loading quantized model from {MODEL_QAT_PATH}...")
    model = keras.models.load_model(MODEL_QAT_PATH, custom_objects=custom_objects)
    model.summary()

    # 3. Create HLS configuration
    config = create_hls_config(model)
    print("\nHLS Configuration (Global):")
    print(config["Model"])

    # 4. Convert model
    print("\nStarting hls4ml conversion...")
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=config["OutputDir"],
        part=config["Part"],
        clock_period=config["ClockPeriod"],
        io_type=config["IOType"],
    )
    print("Conversion complete.")

    # 5. Compile HLS project
    hls_model.write()
    print(f"HLS project generated in {HLS_PROJECT_PATH}")

    # 6. Build the IP (formerly script 4)
    print("\n--- Starting Vitis HLS Synthesis (Build) ---")
    print("This process will synthesize the C++ design into an RTL IP block.")
    print("Note: This may take several minutes to complete.")
    
    report = hls_model.build(
        csim=False, 
        synth=True, 
        cosim=False, 
        export=True, 
        vsynth=True 
    )
    
    print("HLS Build finished.")
    print("\n--- IP Generation Complete ---")
    print("The exported IP can be found in:")
    print(f"{HLS_PROJECT_PATH}/{config['ProjectName']}_prj/solution1/impl/ip")