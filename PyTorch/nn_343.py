# Дана структура нейронной сети (см. на картинке). Такие блоки используются в архитектуре ResNet18 и ResNet34.
# Данная схема отличается от блоков в ResNet только кол-вом каналов.

# В этом упражнении:
# Создайте класс с названием BasicBlock, с помощью которого можно создать нейронную сеть, как на картинке.
# В данной сети входной тензор проходит через два свёрточных слоя, а затем суммируется (skip connection).
# Создайте модель нейронной сети, на вход которой подается тензор с 10-ю каналами, а на выходе получается тензор с 10-ю каналами.
# Результат запишите в переменную block.


import torch
from torch import nn

class BasicBlock(nn.Module):
    def __init__(self, inp_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(inp_channels, 10, (3, 3), padding=1, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Conv2d(10, out_channels, (3, 3), padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.activation = nn.ReLU()

    def forward(self, batch):
        outputs = self.layer(batch) + batch
        return self.activation(outputs)

block = BasicBlock(10, 10)
