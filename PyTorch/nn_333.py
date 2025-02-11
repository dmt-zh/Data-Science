# Дана структура нейронной сети (см. на картинке).
# С помощью класса Sequential создайте модель нейронной сети с тремя последовательными
# свёрточными слоями и функцией активации relu между ними.
# На вход сети будет подаваться тензор с размером - (batch_size, 1, 10, 10)
# Результат запишите в переменную model.


import torch
from torch import nn

model = nn.Sequential(
    nn.Conv2d(1, 5, (3, 3)),
    nn.ReLU(),
    nn.Conv2d(5, 10, (3, 3)),
    nn.ReLU(),
    nn.Conv2d(10, 15, (3, 3))
)
