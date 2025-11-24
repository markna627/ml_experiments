import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
  def __init__(self):
    '''
    batch_n = 400
    batch_size = 100
    H = 32
    W = 32
    C = 3
    F = 5 in the first conv2D layer
    F = 3 in the second conv2D layer
    '''
    super().__init__()

    self.layers = nn.Sequential(
        nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = (3,3), padding = 1), #in_channels , out_channels = # of kernels we want
        # NxCxHxW
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.MaxPool2d(kernel_size = (2,2), stride = 2),

        nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3,3), padding = 1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.MaxPool2d(kernel_size = (2,2), stride = 2),

        nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = (3,3), padding = 1),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.MaxPool2d(kernel_size = (2,2), stride = 2)
    )

    self.head = nn.Sequential(
        nn.Flatten(),
        nn.Linear(128*4*4, 256),
        nn.Dropout(0.25),
        nn.Linear(256, 20),
    )

    self.param_init()
    

  def param_init(self):
    for conv_layer in self.layers:
      if not isinstance(conv_layer, nn.Conv2d):
        continue
      nn.init.kaiming_uniform_(conv_layer.weight, mode = "fan_in", nonlinearity = "relu")
    for conv_head in self.head:
      if not isinstance(conv_head, nn.Linear):
        continue
      nn.init.xavier_uniform_(conv_head.weight)

  def forward(self, x):
    x = self.layers(x)
    return self.head(x)











