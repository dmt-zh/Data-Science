# В этом упражнении:
# Создайте класс с названием MyModel, с помощью которого можно создать нейронную сеть, как на картинке.
# Создайте модель нейронной сети, у которой размер входного и выходного тензора равен 30.
# Результат запишите в переменную model.


import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, features, predictions):
        super().__init__()
        self.activate = nn.ReLU()

        self.layer_1 = self.dense_layer(inner=features, hidden=10, outer=15)
        self.layer_2 = self.dense_layer(inner=15, hidden=10, outer=7)
        self.layer_3 = self.dense_layer(inner=7, hidden=10, outer=15)
        self.layer_4 = self.dense_layer(inner=15, hidden=10, outer=20)
        self.ffn_layer_1 = self.linear_layer(inner=7, outer=7)
        self.ffn_layer_2 = self.linear_layer(inner=20, outer=predictions)

    def linear_layer(self, inner, outer):
        return nn.Linear(inner, outer)

    def dense_layer(self, inner, hidden, outer):
        return nn.Sequential(
            self.linear_layer(inner, hidden),
            self.activate,
            self.linear_layer(hidden, outer),
            self.activate,
        )

    def forward(self, features):
        pass_1 = self.layer_1(features)
        pass_2 = self.layer_2(pass_1)
        ffn_pass_1 = self.activate(self.ffn_layer_1(pass_2))
        pass_3 = self.layer_3(ffn_pass_1 + pass_2)
        pass_4 = self.layer_4(pass_1 + pass_3)
        return self.ffn_layer_2(pass_4)

        
model = MyModel(30, 30)
