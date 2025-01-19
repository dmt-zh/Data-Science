# Дана структура нейронной сети (см. на картинке). Такие блоки используются в архитектуре SqueezeNet.
# Данная схема отличается от блоков в SqueezeNet только кол-вом каналов.

# В этом упражнении:
# Создайте класс с названием Fire, с помощью которого можно создать нейронную сеть, как на картинке.
# При создании данного блока нужно указывать inp_channels, channels_1, channels_2, channels_3.
# На основе созданного класса, создайте один блок, в качестве входных параметров укажите значения - 4, 2, 4, 4.
# Результат запишите в переменную fire.


import torch
from torch import nn

class Fire(nn.Module):
    def __init__(self, inp_c, inp_c1, inp_c2, inp_c3):
        super().__init__()
        self.conv_1 = self._init_conv_layer(inp_c, inp_c1)
        self.conv_2 = self._init_conv_layer(inp_c1, inp_c2, kernel=(3, 3), pad=1)
        self.conv_3 = self._init_conv_layer(inp_c1, inp_c3)

    def _init_conv_layer(self, inp_c, out_c, kernel=(1, 1), pad=0):
        return nn.Sequential(
            nn.Conv2d(inp_c, out_c, kernel_size=kernel, padding=pad),
            nn.ReLU(),
        )

    def forward(self, batch):
        outputs = self.conv_1(batch)
        return torch.cat([self.conv_2(outputs), self.conv_3(outputs)], dim=1)

fire= Fire(4, 2, 4, 4)
