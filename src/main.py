import numpy as np
from brain_builder import BrainBuilder
from layer import Activation, Dense
from loss import Loss
import cv2
import os

def main():
    img0 = np.array([[c/255.0] for c in cv2.imread("images/0.png", 0).flatten()])
    img1 = np.array([[c/255.0] for c in cv2.imread("images/1.png", 0).flatten()])
    img2 = np.array([[c/255.0] for c in cv2.imread("images/2.png", 0).flatten()])
    img3 = np.array([[c/255.0] for c in cv2.imread("images/3.png", 0).flatten()])
    img8 = np.array([[c/255.0] for c in cv2.imread("images/8.png", 0).flatten()])

    xor_x = np.array([[[0], [0]],[[0], [1]],[[1], [0]],[[1], [1]]], dtype=np.float32)
    xor_y = np.array([[0], [1], [1], [0]], dtype=np.float32)

    img_datas = np.array([img0, img1, img2, img3, img8], dtype=np.float32)
    img_labels = np.array([[[0], [0], [0], [0]], [[0], [0], [0], [1]], [[0], [0], [1], [0]], [[0], [0], [1], [1]], [[1], [0], [0], [0]]])

    img_gen_datas = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 1, 1]], dtype=np.float32)
    img_gen_labels = np.array([img0, img1, img2, img3, img8])

    xor_brain = BrainBuilder([
        Dense(2, 2),
        Activation(Activation.ActivationType.Sigmoid),
        Dense(2, 1),
        Activation(Activation.ActivationType.Sigmoid)
    ], np.float32(3), Loss.LossType.MSE)

    image_brain = BrainBuilder([
        Dense(28*28, 28),
        Activation(Activation.ActivationType.Sigmoid),
        Dense(28, 5),
        Activation(Activation.ActivationType.Sigmoid),
        Dense(5, 4),
        Activation(Activation.ActivationType.Sigmoid)
    ], np.float32(0.1), Loss.LossType.MSE)

    image_gen_brain = BrainBuilder([
        Dense(4, 8),
        Activation(Activation.ActivationType.Sigmoid),
        Dense(16, 100),
        Activation(Activation.ActivationType.Sigmoid),
        Dense(100, 200),
        Activation(Activation.ActivationType.Sigmoid),
        Dense(200, 28*28),
        Activation(Activation.ActivationType.Sigmoid)
    ], np.float32(10), Loss.LossType.MSE)

    #xor_brain.train(1000, xor_x, xor_y)
    #xor_brain.predict(xor_x, xor_y)

    image_brain.train(1000, img_datas, img_labels)
    image_brain.predict(img_datas, img_labels)

if __name__ == "__main__":
    main()