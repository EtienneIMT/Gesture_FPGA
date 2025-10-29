import torch.nn as nn
import torch.nn.functional as F
# Importer Brevitas
import brevitas.nn as qnn 
from brevitas.quant import Int8WeightPerTensorFloat, Int8ActPerTensorFloat, Int8Bias, Uint8ActPerTensorFloat

class GestureNet(nn.Module):
    def __init__(self, num_classes=5):
        super(GestureNet, self).__init__()

        # 1. Couche de quantification d'entrée
        self.quant_inp = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat, # Quantifie l'entrée en INT8
            return_quant_tensor=True
        )
        
        # --- Couches Quantifiées (INT8) ---
        self.conv1 = qnn.QuantConv2d(
            in_channels=1, 
            out_channels=16, 
            kernel_size=3, 
            padding=1,
            weight_quant=Int8WeightPerTensorFloat, # Poids INT8
            bias=True,
            bias_quant=Int8Bias # Biais INT8
        )
        self.relu1 = qnn.QuantReLU(
            act_quant=Uint8ActPerTensorFloat, # Activation UINT8 (car INT8 pas optimisé avec ReLU)
            return_quant_tensor=True # Important pour passer au suivant
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = qnn.QuantConv2d(
            in_channels=16, 
            out_channels=32, 
            kernel_size=3, 
            padding=1,
            weight_quant=Int8WeightPerTensorFloat,
            bias=True,
            bias_quant=Int8Bias
        )
        self.relu2 = qnn.QuantReLU(
            act_quant=Uint8ActPerTensorFloat,
            return_quant_tensor=True
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        
        # Taille après 2 poolings sur du 64x64 -> 16x16
        self.fc1 = qnn.QuantLinear(
            32 * 16 * 16, 
            128,
            weight_quant=Int8WeightPerTensorFloat,
            bias=True,
            bias_quant=Int8Bias
        )
        self.relu3 = qnn.QuantReLU(
            act_quant=Uint8ActPerTensorFloat,
            return_quant_tensor=True
        )
        
        # Couche finale (souvent non quantifiée ou quantifiée différemment)
        self.fc2 = qnn.QuantLinear(
            128, 
            num_classes,
            weight_quant=Int8WeightPerTensorFloat,
            bias=True,
            bias_quant=Int8Bias
        )

    def forward(self, x):
        # L'activation d'entrée doit aussi être quantifiée
        # (On peut ajouter un qnn.QuantIdentity() ici, 
        # mais Brevitas peut le gérer à l'export)

        # Applique la quantification d'entrée D'ABORD
        x = self.quant_inp(x)
        
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.flatten(x)
        
        x = self.fc1(x)
        x = self.relu3(x)
        
        x = self.fc2(x)
        return x