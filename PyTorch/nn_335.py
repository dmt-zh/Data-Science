# В этом упражнении:
# Создайте класс с названием MyModel, с помощью которого можно создать нейронную сеть, как на картинке.
# Конкатенация производится по каналам.
# Создайте модель нейронной сети, на вход которой подается тензор с размером - (batch_size, 5, 10, 10).
# Результат запишите в переменную model.
# Рассчитайте размер тензоров на выходе с левого свёрточного слоя, с правого свёрточного слоя и после конкатенации.
# При условии, что на вход нейронной сети подается тензор размером - (16, 5, 10, 10). 
# Результат запишите в виде кортежа в соответствующие переменные left_conv, right_conv и out_concat.


import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, inp_channels):
        super().__init__()
        self.left_conv_layer = nn.Conv2d(inp_channels, 10, (1, 1))
        self.right_conv_layer = nn.Conv2d(inp_channels, 10, (3, 3), padding=1)
        self.activation = nn.ReLU()

    def forward(self, batch):
        left_conv = self.activation(
            self.left_conv_layer(batch)
        )
        right_conv = self.activation(
            self.right_conv_layer(batch)
        )
        return torch.cat([left_conv, right_conv], dim=1)


batch = torch.rand([16, 5, 10, 10])
model = MyModel(5)
left_conv = model.left_conv_layer(batch).shape
right_conv = model.right_conv_layer(batch).shape
out_concat = model(batch).shape
