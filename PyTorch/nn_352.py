# В этом упражнении:
# Создайте класс с названием Transition, с помощью которого можно создать нейронную сеть, как на картинке.
# Создайте модель нейронной сети, на вход которой подается тензор с 3-я каналами, а на выходе получается 
# тензор с 5-ю каналами. Результат запишите в переменную transition.


import torch
from torch import nn

class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, (1, 1), bias=False),
            nn.AvgPool2d((2, 2), stride=2)
        )

    def forward(self, batch):
        return self.conv_layer(batch)

transition = Transition(3, 5)
