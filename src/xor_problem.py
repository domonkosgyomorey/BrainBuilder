import numpy as np
from brain_builder import BrainBuilder
from layer import Activation, Dense
from loss import Loss

xor_x = np.array([[0, 0],[0, 1],[1, 0],[1, 1]], dtype=np.float32)
xor_y = np.array([[0], [1], [1], [0]], dtype=np.float32)

xor_brain = BrainBuilder([
        Dense(2, 2),
        Activation(Activation.ActivationType.Sigmoid),
        Dense(2, 1),
        Activation(Activation.ActivationType.Sigmoid)
    ], np.float32(5), Loss.LossType.MSE)

xor_brain.train(3500, xor_x, xor_y)
xor_brain.predict(xor_x, xor_y)