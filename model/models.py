import numpy as np
import model.function as F
import model.submodule as nn
from model.submodule import baseLayer
from pathlib import Path

class CNN(baseLayer):
    def __init__(self):
        pass

class mlp_leaky(baseLayer):
    def __init__(self):
        pass

class mlp(baseLayer):
    def __init__(self):
        super().__init__()
        self.sequntial = [
            nn.linear(784, 300),
            nn.ReLU(),
            nn.linear(300, 300),
            nn.ReLU(),
            nn.linear(300, 10)
        ]
        self.activation = []

    def forward(self, x):
        self.activation = list()
        self.activation.append(x)
        for layer in self.sequntial:
            x = layer(x)
            self.activation.append(x)
        self.activation.pop(-1)
        return x

    def update(self, grad_output):
        rev_activation = list(reversed(self.activation))
        for index, layer in enumerate(reversed(self.sequntial)):
            grad_output = layer.backward(rev_activation[index], grad_output)

    def save(self, path):
        dict = {}
        for i, layer in enumerate(self.sequntial):
            if layer.has_param:
                dict['weight{:1d}'.format(i)] = layer.weight
                dict['bias{:1d}'.format(i)] = layer.bias
        np.save(path, dict)

    def load(self, path):
        path = Path(path)
        dict = np.load(path, allow_pickle=True).item()
        for i, layer in enumerate(self.sequntial):
            if layer.has_param:
                layer.weight = dict['weight{:1d}'.format(i)]
                layer.bias = dict['bias{:1d}'.format(i)]


