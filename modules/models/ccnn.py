import torch.nn as nn

class CCNN(nn.Module): # Cross-Correlation-Neural-Network
    def __init__(self, kernel_size, num_layers):
        super().__init__()

        self.layers = nn.ModuleList() # Its like: layers = [], definition of empty list
        self.layers.append(nn.Conv2d(1, 64, kernel_size, padding=kernel_size // 2)) # 1 image input -> 64 resulting featuremaps
        self.layers.append(nn.PReLU()) # activationlayer

        for _ in range(num_layers - 2):
            self.layers.append(nn.Conv2d(64, 64, kernel_size, padding=kernel_size // 2)) # 64 featuremaps input -> 64 images output per layer
            self.layers.append(nn.PReLU()) #activationlayer

        self.final = nn.Conv2d(64, 1, kernel_size, padding=kernel_size // 2) # 64 featuremaps input -> final image output

    def forward(self, x): # jump from layer to layer in self.layers
        for layer in self.layers:
            x = layer(x)
        return self.final(x)
