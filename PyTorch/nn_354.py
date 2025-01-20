# В этом упражнении:
# Создайте класс с названием MyModel, с помощью которого можно создать нейронную сеть, как на картинке.
# На вход модели будут подаваться тензоры размером - (batch_size, 5, 10, 10),
# а на выходе должны получаться тензоры размером - (batch_size, 5)
# На основе класса MyModel создайте модель нейронной сети. Результат запишите в переменную model


import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, in_channels, out_dim):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels, 10, (3, 3), bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.AvgPool2d((8, 8))
        )
        self.linear_layer = nn.Linear(10, out_dim)

    def forward(self, batch):
        outputs = self.conv_layer(batch)
        flattened = outputs.flatten(start_dim=1, end_dim=-1)
        return self.linear_layer(flattened)


model = MyModel(5, 5)
