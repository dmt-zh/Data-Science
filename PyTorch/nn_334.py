# В этом упражнении:
# Создайте класс с названием MyModel, с помощью которого можно создать нейронную сеть, как на картинке.
# Создайте модель нейронной сети, у которой один канал у входного тензора и 15 у выходного.
# Результат запишите в переменную model.


import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, inp_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(inp_channels, 5, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(5, 10, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(10, out_channels, (3, 3))
        )

    def forward(self, batch):
        return self.layer(batch)

model = MyModel(1, 15)
