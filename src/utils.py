import cv2
import numpy as np

def read_mono_img_norm(image_path):
    return np.array([[c/255.0] for c in cv2.imread(image_path, 0).flatten()])

# xs: [0, 0] => [[0], [0]]
def create_labels(xs):
    res = []
    for x in xs:
        res.append(np.array(x).reshape(len(x), 1))
    return res

# xs: [[0, 0], [0, 0]] => [[[0], [0]], [[0], [0]]]
def create_datas(xs):
    res = []
    for x in xs:
        res.append(np.array(x).T)
    return res