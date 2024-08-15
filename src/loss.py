from enum import Enum
import numpy as np


def mse(predicted: np.ndarray, output: np.ndarray):
    return np.mean((predicted - output) ** 2)


def mse_derivative(predicted: np.ndarray, output: np.ndarray):
    return 2.0 * (predicted - output) / float(predicted.size)


class Loss:

    class LossType(Enum):
        MSE=1,

    def __init__(self, loss_type: LossType) -> None:
        match loss_type:
            case Loss.LossType.MSE:
                self._loss_fn = mse
                self._loss_fn_der = mse_derivative
            case _:
                raise Exception("Unexpected loss function type: ", loss_type)

    def get_loss(self, predicted, output):
        return self._loss_fn(predicted, output)
    
    def get_loss_derivative(self, predicted, output):
        return self._loss_fn_der(predicted, output)
