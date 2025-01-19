# В этом упражнении:
# Создайте класс Bottleneck со слоем downsample, реализованный в одном из прошлых упражнении.
# Создайте класс с названием MyModel, с помощью которого можно создать нейронную сеть, как на картинке справа.
# На вход модели будут подаваться тензоры размером - (batch_size, 1, 10, 10), на выходе с Bottleneck слоя будет 1 канал,
# а на выходе сети будут векторы с 5-ю элементами.
# На основе класса MyModel создайте модель нейронной сети. Результат запишите в переменную model.


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


class MyModel(nn.Module):
    def __init__(self, inp_channels, out_dim):
        super().__init__()
        self.bottleneck_layer = Bottleneck(1, 1)
        self.linear_layer = nn.Sequential(
            nn.Linear(inp_channels * 5 * 5, 7),
            nn.ReLU(),
            nn.Linear(7, out_dim)
        )

    def forward(self, batch):
        outputs = self.bottleneck_layer(batch)
        flattened = outputs.flatten(start_dim=1, end_dim=-1)
        return self.linear_layer(flattened)

model = MyModel(1, 5)
