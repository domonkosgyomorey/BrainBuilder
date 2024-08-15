import numpy as np
from layer import ILayer
from loss import Loss 

class BrainBuilder:
    
    def __init__(self, layers: list[ILayer], learning_rate: np.float32, loss_type: Loss.LossType) -> None:
        self._layers = layers
        self._loss = Loss(loss_type)
        self._lr = learning_rate

    def feedforward(self, xs: np.ndarray):
        layer_input = np.array(xs).reshape(xs.size, 1)
        for layer in self._layers:
            layer_input = layer.feedforward(layer_input)
        return layer_input

    def get_loss(self, predicted, output):
        return self._loss.get_loss(predicted, output)

    def learn(self, predicted: np.ndarray, output: np.ndarray):
        dn = self._loss.get_loss_derivative(predicted, output)
        xi = dn
        for layer in self._layers[::-1]:
            xi = layer.learn(self._lr, xi)

    def train(self, max_iter, datas, labels):
        i = 0
        loss = 1000
        while i < max_iter:
            for j in range(len(datas)):
                prediction = self.feedforward(datas[j])
                loss = (self.get_loss(prediction, labels[j]))
                self.learn(prediction, labels[j])
            
            if i % 100 == 0:
                print(f"({i}/{max_iter}) iteration, Loss: {loss}")
            i += 1

    def predict(self, datas, labels):
        for i, data in enumerate(datas):
            res = self.feedforward(data)
            print(labels[i].T, " => ", res.T)