# Дана структура нейронной сети (см. на картинке).

# В этом упражнении:
# Создайте класс с названием MyModel, с помощью которого можно создать нейронную сеть, как на картинке.
# В данной модели каждый последующий линейный слой имеет на один нейрон меньше предыдущего. При создании класса MyModel воспользуйтесь классом nn.ModuleList.
# Создайте модель нейронной сети, у которой размер входного тензора равен 20, а выходного 10. Результат запишите в переменную model.


import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, features, predictions):
        super().__init__()
        self.layers = nn.ModuleList()
        for layer_out in range(19, 10, -1):
            self.layers.append(nn.Linear(features, layer_out))
            self.layers.append(nn.ReLU())
            features = layer_out
        self.layers.append(nn.Linear(features, predictions))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

model = MyModel(20, 10)
