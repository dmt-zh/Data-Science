import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, features, state_1, state_2, predictions):
        super().__init__()
        self.dense_layer_1 = self.layer_wrapper(embedding=features)
        self.dense_layer_2 = self.layer_wrapper()
        self.ffn_layer = nn.Linear(15, predictions)
        self.activated_state_1 = self.layer_wrapper(num_units=state_1)
        self.activated_state_2 = self.layer_wrapper(num_units=state_2)

    def layer_wrapper(self, embedding=15, num_units=15):
        return nn.Sequential(
            nn.Linear(embedding, 10),
            nn.ReLU(),
            nn.Linear(10, num_units),
            nn.ReLU()
        )

    def forward(self, features):
        extras_1 = self.dense_layer_1(features)
        extras_2 = self.dense_layer_2(extras_1)
        predicted_out = self.ffn_layer(extras_2)
        out_1 = self.activated_state_1(extras_1)
        out_2 = self.activated_state_2(extras_2)
        return [out_1, out_2, predicted_out]

model = MyModel(20, 20, 20, 10)
