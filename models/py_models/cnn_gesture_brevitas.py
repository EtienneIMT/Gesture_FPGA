import torch.nn as nn
import torch.nn.functional as F
import brevitas.nn as qnn 
from brevitas.quant import (
    Int8WeightPerTensorFloat, 
    Int8ActPerTensorFloat, 
    Int8Bias, 
    Uint8ActPerTensorFloat
)

class GestureNet(nn.Module):
    def __init__(self, num_classes=5):
        super(GestureNet, self).__init__()

        self.quant_inp = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat, # Entrée signée (-1 à 1)
            return_quant_tensor=True
        )
        
        # --- Bloc Conv 1 ---
        self.conv1 = qnn.QuantConv2d(
            in_channels=1, out_channels=16, kernel_size=3, padding=1,
            weight_quant=Int8WeightPerTensorFloat, bias=True, bias_quant=Int8Bias)
        self.relu1 = qnn.QuantReLU(
            act_quant=Uint8ActPerTensorFloat, # Sortie NON SIGNÉE
            return_quant_tensor=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Couche standard, transmet UINT8

        # --- Bloc Conv 2 ---
        # L'entrée de conv2 est la sortie de pool1 (donc UINT8)
        # Brevitas gère cela automatiquement
        self.conv2 = qnn.QuantConv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1,
            weight_quant=Int8WeightPerTensorFloat, bias=True, bias_quant=Int8Bias)
        self.relu2 = qnn.QuantReLU(
            act_quant=Uint8ActPerTensorFloat, # Sortie NON SIGNÉE
            return_quant_tensor=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Couche standard, transmet UINT8

        # --- Couche Flatten ---
        self.flatten = nn.Flatten()

        # --- Couches FC ---
        # L'entrée de fc1 est la sortie de pool2 (donc UINT8)
        self.fc1 = qnn.QuantLinear(
            32 * 16 * 16, 128,
            weight_quant=Int8WeightPerTensorFloat, bias=True, bias_quant=Int8Bias)
        self.relu3 = qnn.QuantReLU(
            act_quant=Uint8ActPerTensorFloat, # Sortie NON SIGNÉE
            return_quant_tensor=True)

        # Couche de sortie
        self.fc2 = qnn.QuantLinear(
            128, num_classes,
            weight_quant=Int8WeightPerTensorFloat, bias=True, bias_quant=Int8Bias)

    def forward(self, x):
        x = self.quant_inp(x)
        
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        # Pas besoin de quant_ident1

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        # Pas besoin de quant_ident2

        x = self.flatten(x)
        
        x = self.fc1(x)
        x = self.relu3(x)
        
        x = self.fc2(x)
        return x