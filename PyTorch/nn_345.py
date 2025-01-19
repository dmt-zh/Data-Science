# В этом упражнении:
# Создайте класс с названием Bottleneck, с помощью которого можно создать нейронную сеть, как на картинке. 
# Создайте модель нейронной сети, на вход которой подается тензор с 5-ю каналами, а на выходе получается тензор с 5-ю каналами.
# Результат запишите в переменную block.
# Проанализируйте как меняется высота и ширина входного тензора после каждого свёрточного слоя.


import torch
from torch import nn

class Bottleneck(nn.Module):
    def __init__(self, inp_channels, out_channels):
        super().__init__()
        self.main_layer = nn.Sequential(
            nn.Conv2d(inp_channels, 7, (1, 1), bias=False),
            nn.BatchNorm2d(7),
            nn.ReLU(),
            nn.Conv2d(7, 7, (3, 3), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(7),
            nn.ReLU(),
            nn.Conv2d(7, out_channels, (1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.branch_layer = nn.Sequential(
            nn.Conv2d(inp_channels, out_channels, (1, 1), stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.activation = nn.ReLU()

    def forward(self, batch):
        outputs = self.main_layer(batch) + self.branch_layer(batch)
        return self.activation(outputs)

block = Bottleneck(5, 10)
