# В этом упражнении:
# Создайте класс с названием MyModel, с помощью которого можно создать нейронную сеть, как на картинке.
# На вход модели будут подаваться тензоры размером - (batch_size, 3, 10, 10), а на выходе будут векторы с 10-ю элементами.
# На основе класса MyModel создайте модель нейронной сети. Результат запишите в переменную model.


import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, inp_channels, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.layer = nn.Sequential(
            nn.Conv2d(inp_channels, 5, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(5, 5, (3, 3), padding=1),
            nn.ReLU(),
        )
        self.linear_layer = nn.Linear(5*8*8, out_dim)

    def forward(self, batch):
        outputs = self.layer(batch)
        flattened = outputs.flatten(start_dim=1, end_dim=-1)
        return self.linear_layer(flattened)

model = MyModel(3, 10)
