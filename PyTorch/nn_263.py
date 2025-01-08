# В этом упражнении:
# Создайте класс с названием MyModel, с помощью которого можно создать нейронную сеть, как на картинке.
# Создайте модель нейронной сети, у которой размер входного тензора равен inp=20, а размеры выходных 
# тензоров равны out_1 15 и out_2 15.  Результат запишите в переменную model.
# Нейронная сеть должна возвращать список с out_1 и out_2.


import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, features, hidden_state, predictions):
        super().__init__()
        self.dense_layer_1 = nn.Sequential(
            nn.Linear(features, 10),
            nn.ReLU(),
            nn.Linear(10, hidden_state)
        )
        self.dense_layer_2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_state, 10),
            nn.ReLU(),
            nn.Linear(10, predictions)
        )

    def forward(self, features):
        outputs = []
        extra_output = self.dense_layer_1(features)
        predicted_out = self.dense_layer_2(extra_output)
        outputs.append(extra_output)
        outputs.append(predicted_out)
        return outputs

model = MyModel(20, 15, 15)
