# В этом упражнении:
# Создайте класс с названием MyModel, с помощью которого можно создать нейронную сеть, как на картинке.
# Создайте модель нейронной сети, на вход которой подается тензор с 5-ю каналами, а на выходе получается
# тензор с 10-ю каналами. Результат запишите в переменную model.


import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, inp_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(inp_channels, out_channels, (3, 3), padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, batch):
        return self.layer(batch)

model = MyModel(5, 10)
