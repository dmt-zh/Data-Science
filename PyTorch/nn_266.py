# В этом упражнении:
# Создайте класс с названием MyModel, с помощью которого можно создать нейронную сеть, как на картинке.
# Создайте модель нейронной сети на основе класса MyModel, с помощью которой можно обучить классификатор на имеющемся наборе данных.
# Изображения в нейронную сеть подаются в виде вектора.
# Результат запишите в переменную model.
# Создайте функцию потерь для нейронной сети и запишите ее в переменную loss_model.
# В качестве оптимизатора градиентного спуска выберите Adam со скоростью обучения равной 0.01.
# Оптимизатор запишите в переменную opt.


import torch
from torch import nn
from torch.optim import Adam

class MyModel(nn.Module):
    def __init__(self, inner_dim, outer_dim):
        super().__init__()
        self.tranform = nn.Sequential(
            nn.Linear(inner_dim, 52),
            nn.ReLU(),
            nn.Linear(52, 26),
            nn.ReLU(),
            nn.Linear(26, outer_dim),
        )
        self._dim = inner_dim

    def forward(self, batch):
        flatten = batch.view([-1, self._dim])
        return self.tranform(flatten)

model = MyModel(32*32, 50)
loss_model = nn.CrossEntropyLoss()
opt = Adam(model.parameters(), lr=0.01)
