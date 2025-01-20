# В этом упражнении:
# Создайте класс с названием MyModel, с помощью которого можно создать нейронную сеть, как на картинке.
# На вход модели будут подаваться тензоры размером - (batch_size, 5, 10, 10), а на выходе должны получаться
# тензоры размером - (batch_size, 10, 1, 1)
# Создайте модель нейронной сети, на вход которой подается тензор с 5-ю каналами, а на выходе получается тензор
# с 10-ю каналами. Результат запишите в переменную model.


import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.AvgPool2d((out_channels, out_channels))
        )

    def forward(self, batch):
        return self.conv_layer(batch)

model = MyModel(5, 10)
