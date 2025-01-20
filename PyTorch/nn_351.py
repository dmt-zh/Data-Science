# В этом упражнении:
# Создайте класс с названием MyModel, с помощью которого можно создать нейронную сеть, как на картинке.
# Создайте модель нейронной сети, на вход которой подается тензор с 3-я каналами, а на выходе получается
# тензор с 5-ю каналами. Результат запишите в переменную model.


import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels, 5, (7, 7), stride=2, padding=3, bias=False),
            nn.BatchNorm2d(5),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), stride=2, padding=1)
        )

    def forward(self, batch):
        return self.conv_layer(batch)

model = MyModel(3, 5)
