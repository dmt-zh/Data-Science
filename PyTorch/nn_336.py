# В этом упражнении:
# Создайте класс с названием MyModel, с помощью которого можно создать нейронную сеть, как на картинке.
# На вход модели подается список с тензорами, размеры которых отличаются только количеством каналов. Конкатенация производится по каналам.
# Создайте модель нейронной сети, на вход которой подается список с четырьмя тензорами.
# Кол-во каналов в тензорах равны - 4, 3, 3 и 3. На выходе должен получится тензор с 10-ю каналами. Результат запишите в переменную model.


import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, inp_channels, out_channels):
        super().__init__()
        self.conv_layer = nn.Conv2d(inp_channels, out_channels, (3, 3))
        self.activation = nn.ReLU()

    def forward(self, batch):
        outputs = self.conv_layer(torch.cat(batch, dim=1))
        return self.activation(outputs)

model = MyModel(13, 10)
