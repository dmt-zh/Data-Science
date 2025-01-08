# В этом упражнении:
# Создайте класс BasicBlock , реализованный в прошлом упражнении.
# Создайте класс с названием MyModel, с помощью которого можно создать нейронную сеть, как на картинке справа.
# Она состоит из 10 последовательных блоков класса BasicBlock и линейного слоя в конце.
# На основе класса MyModel создайте модель нейронной сети, у которой размер входного тензора равен 5, а выходного 15.
# Результат запишите в переменную model.


import torch
from torch import nn

class BasicBlock(nn.Module):
    def __init__(self, features, predictions):
        super().__init__()
        self._layer = nn.Sequential(
            nn.Linear(features, 10),
            nn.ReLU(),
            nn.Linear(10, predictions),
        )
        self._activate = nn.ReLU()

    def forward(self, x):
        return self._activate(self._layer(x) + x)

class MyModel(nn.Module):
    def __init__(self, features, predictions, num_layers=10):
        super().__init__()
        self.encoder = nn.Sequential()
        for num in range(num_layers):
            self.encoder.add_module(f'{num}', BasicBlock(features, features))
        self.linear_layer = nn.Linear(features, predictions)

    def forward(self, features):
        encoded_features = self.encoder(features)
        return self.linear_layer(encoded_features)
    
model = MyModel(5, 15)
batch = torch.rand([20, 5], dtype=torch.float32)
encoded = model(batch)
print(f'Encoded tensor shape: {encoded.shape}')
