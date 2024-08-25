from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSlider, QFileDialog
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt
import numpy as np
import cv2
from PIL import Image as PILImage
from brain_builder import BrainBuilder
from layer import Dense, Activation
from loss import Loss
from utils import read_mono_img_norm
from learning_rate import LearningRateSchedule

class ImageTransformApp(QWidget):
    def __init__(self):
        super().__init__()

        self.xs = []
        self.ys = []
        for line in open("./datas/img_gen.txt").readlines():
            x, y = line.split(' ')
            for i in range(10):
                path = y.replace('-', str(i)).strip()
                print(x, " -> ", path, " loaded")
                self.xs.append(int(x))
                self.ys.append(np.array(read_mono_img_norm(path)))


        self.xs = np.array(self.xs)
        self.ys = np.array(self.ys)
        
        self.brain = BrainBuilder([
            Dense(1, 50),
            Activation(Activation.ActivationType.LeakyReLU),
            Dense(50, 70),
            Activation(Activation.ActivationType.LeakyReLU),
            Dense(70, 100),
            Activation(Activation.ActivationType.LeakyReLU),
            Dense(100, 100),
            Activation(Activation.ActivationType.TanH),
            Dense(100, 784),
            Activation(Activation.ActivationType.Sigmoid)
        ],0.1, Loss.LossType.MAE, True, LearningRateSchedule.STEP_DECAY)

        self.max_iter = 10000

        self.initUI()

    def initUI(self):
        main_layout = QHBoxLayout()
        
        tool_layout = QVBoxLayout()
        tool_layout.setSpacing(10)
        tool_layout.setContentsMargins(20, 20, 20, 20)

        slider_layout = QHBoxLayout()
        self.trans_slider = QSlider(Qt.Horizontal)
        self.trans_slider.setRange(0, 9000)
        self.trans_slider.setValue(0)
        self.trans_slider.setTickInterval(1)
        self.trans_slider.setTickPosition(QSlider.TicksBelow)
        self.trans_slider.setSingleStep(1)
        self.trans_slider.valueChanged.connect(self.re_calculate_image)

        
        self.trans_label = QLabel(f"{self.trans_slider.value() / 1000.0:.2f}")
        self.trans_label.setFont(QFont("Arial", 12))
        self.trans_label.setStyleSheet("color: #3333FF;")
        
        slider_layout.addWidget(self.trans_slider)
        slider_layout.addWidget(self.trans_label)
        tool_layout.addLayout(slider_layout)

        btn_font = QFont("Arial", 12, QFont.Bold)
        
        style = """
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px;
            border-radius: 5px;
        """

        self.train_btn = QPushButton('Train')
        self.train_btn.setFont(btn_font)
        self.train_btn.setStyleSheet(style)
        self.train_btn.clicked.connect(self.train)
        
        tool_layout.addWidget(self.train_btn)

        main_layout.addLayout(tool_layout)

        self.out_img_label = QLabel()
        self.out_img_label.setAlignment(Qt.AlignCenter)
        self.update_image()

        main_layout.addWidget(self.out_img_label)

        self.setLayout(main_layout)

        self.setWindowTitle('Image Playground')
        self.resize(900, 600)

    def train(self):
        self.brain.train(10000, self.xs, self.ys)
        self.re_calculate_image()

    def re_calculate_image(self):
        value = self.trans_slider.value() / 1000.0
        img = (self.brain.feedforward(np.array([value]).reshape(1, 1)).reshape((28, 28))*255.0).astype(np.uint8)
        cv2.imwrite("out.png", img)
        self.update_image()
        self.trans_label.setText(f"{value:.2f}")

    def update_image(self):
        pil_img = PILImage.open("out.png")
        qimage = QImage(pil_img.tobytes(), pil_img.width, pil_img.height, pil_img.width, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)
        self.out_img_label.setPixmap(pixmap.scaled(self.out_img_label.size(), Qt.KeepAspectRatio))

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    ex = ImageTransformApp()
    ex.show()
    sys.exit(app.exec_())
