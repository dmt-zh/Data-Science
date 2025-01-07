# Дана структура нейронной сети (см. на картинке).

# В этом упражнении:
# Создайте класс с названием BasicBlock, с помощью которого можно создать нейронную сеть, как на картинке. 
# В данной сети входной тензор проходит через два линейных слоя, а затем суммируется с самим собой.
# Создайте модель нейронной сети, у которой размеры входного и выходного тензора совпадают и равны 20.
#  Результат запишите в переменную block.


import torch
from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, features, predictions):
        super().__init__()
        self._layer = nn.Sequential(
            nn.Linear(features, 10),
            nn.ReLU(),
            nn.Linear(10, predictions),
        )
        self._activate = nn.ReLU()

    def forward(self, x):
        return self._activate(self._layer(x) + x)

block = BasicBlock(20, 20)
