import numpy as np
from brain_builder import BrainBuilder
from layer import Activation, Dense
from loss import Loss
import utils

img_datas = []
img_labels = []
lines = open(file="src/number_recognition_data.txt", mode="r").readlines()
num_of_bit = int(lines.pop(0))
for data in lines:
    split = data.split(' ')
    img_name = split[0]
    label = []
    for b in range(num_of_bit):
        label.append([int(split[b+1])])
    img_labels.append(label)
    img_datas.append(utils.read_mono_img_norm(f"images/{img_name}"))

img_labels = np.array(img_labels)
img_datas = np.array(img_datas)

image_brain = BrainBuilder([
        Dense(28*28, 28),
        Activation(Activation.ActivationType.Sigmoid),
        Dense(28, 5),
        Activation(Activation.ActivationType.Sigmoid),
        Dense(5, 4),
        Activation(Activation.ActivationType.Sigmoid)
    ], np.float32(0.1), Loss.LossType.MSE)

image_brain.train(1000, img_datas, img_labels)
image_brain.predict(img_datas, img_labels)