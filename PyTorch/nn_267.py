# В этом упражнении:
# Создайте класс с названием MyModel, с помощью которого можно создать нейронную сеть, как на картинке. 
# Под операцией суммирования подразумевается конкатенация векторов.
# Создайте модель нейронной сети на основе класса MyModel, с помощью которой можно предсказывать ширину и высоту
#  нужного объекта на изображении. Помимо изображения в нейронную сеть передается дополнительная информация, \
#  закодированная в вектор размером 10. Изображения в нейронную сеть также подаются в виде вектора.
# ​​Созданную модель запишите в переменную model.
# Создайте функцию потерь для нейронной сети и запишите ее в переменную loss_model.
# В качестве оптимизатора градиентного спуска выберите Adam со скоростью обучения равной 0.1. Оптимизатор запишите в переменную opt.


import torch
from torch import nn
from torch.optim import Adam

class MyModel(nn.Module):
    def __init__(self, image_dim, inner_dim, outer_dim):
        super().__init__()
        self.layer_1 = self.dense_layer(image_dim)
        self.layer_2 = self.dense_layer(inner_dim + 10)
        self.ffn_layer = nn.Linear(10, outer_dim)

    def dense_layer(self, inner):
        return nn.Sequential(
            nn.Linear(inner, 10),
            nn.ReLU(),
        )

    def forward(self, image_batch, extras_batch):
        enriched = torch.cat(
            [self.layer_1(image_batch), extras_batch],
            dim=1
        )
        encoded = self.layer_2(enriched)
        return self.ffn_layer(encoded)

model = MyModel(15 * 15, 10, 2)
loss_model = nn.MSELoss()
opt = Adam(model.parameters(), lr=0.1)
