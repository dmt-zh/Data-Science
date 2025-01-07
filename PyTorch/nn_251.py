# Дана структура нейронной сети (см. на картинке).
# В этом упражнении:
# Создайте класс с названием MyModel, с помощью которого можно создать нейронную сеть, как на картинке.
# Создайте модель нейронной сети, у которой размеры входных тензоров равны inp1=124, inp2=124, а выходного 18.
# Результат запишите в переменную model.


import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, features, labels, predictions):
        super().__init__()
        self._features_seq = self._init_seq_model(features)
        self._labels_seq = self._init_seq_model(labels)
        self._linear_out = nn.Linear(52, predictions)

    def _init_seq_model(self, batch):
        seq_model = nn.Sequential(
            nn.Linear(batch, 52),
            nn.ReLU(),
            nn.Linear(52, 26),
            nn.ReLU(),
        )
        return seq_model

    def forward(self, x, y):
        x = self._features_seq(x)
        y = self._labels_seq(y)
        x_out = torch.cat((x, y), dim=1)
        return self._linear_out(x_out)

model = MyModel(124, 124, 18)
