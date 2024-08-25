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

        self.brain = BrainBuilder([
            Dense(1, 5),
            Activation(Activation.ActivationType.Sigmoid),
            Dense(5, 50),
            Activation(Activation.ActivationType.Sigmoid),
            Dense(50, 20),
            Activation(Activation.ActivationType.Sigmoid),
            Dense(20, 784),
            Activation(Activation.ActivationType.Sigmoid)
        ], 7, Loss.LossType.LOG_LOSS, True, LearningRateSchedule.STEP_DECAY)

        self.img_from = np.array(read_mono_img_norm("./images/0_0.png"))
        self.img_to = np.array(read_mono_img_norm("./images/8_0.png"))
        self.max_iter = 3000

        self.initUI()

    def initUI(self):
        main_layout = QHBoxLayout()
        
        tool_layout = QVBoxLayout()
        tool_layout.setSpacing(10)
        tool_layout.setContentsMargins(20, 20, 20, 20)

        slider_layout = QHBoxLayout()
        self.trans_slider = QSlider(Qt.Horizontal)
        self.trans_slider.setRange(0, 1000)
        self.trans_slider.setValue(0)
        self.trans_slider.setTickInterval(1)
        self.trans_slider.setTickPosition(QSlider.TicksBelow)
        self.trans_slider.setSingleStep(1)
        self.trans_slider.valueChanged.connect(self.re_calculate_image)

        
        self.trans_label = QLabel(f"{self.trans_slider.value() / float(self.trans_slider.maximum()):.2f}")
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
        
        self.img_from_btn = QPushButton('Change From Image')
        self.img_from_btn.setFont(btn_font)
        self.img_from_btn.setStyleSheet(style)
        self.img_from_btn.clicked.connect(self.change_from_img)
        
        self.img_to_btn = QPushButton('Change To Image')
        self.img_to_btn.setFont(btn_font)
        self.img_to_btn.setStyleSheet(style)
        self.img_to_btn.clicked.connect(self.change_to_img)
        
        tool_layout.addWidget(self.train_btn)
        tool_layout.addWidget(self.img_from_btn)
        tool_layout.addWidget(self.img_to_btn)

        main_layout.addLayout(tool_layout)

        self.out_img_label = QLabel()
        self.out_img_label.setAlignment(Qt.AlignCenter)
        self.update_image()

        main_layout.addWidget(self.out_img_label)

        self.setLayout(main_layout)

        self.setWindowTitle('Image Playground')
        self.resize(900, 600)

    def train(self):
        self.brain.train(self.max_iter, np.array([0, 1]).reshape(2, 1), [self.img_from, self.img_to])
        self.re_calculate_image()

    def change_from_img(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Image File', '', 'Image Files (*.png)')
        if file_name:
            self.img_from = np.array(read_mono_img_norm(file_name))

    def change_to_img(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Image File', '', 'Image Files (*.png)')
        if file_name:
            self.img_to = np.array(read_mono_img_norm(file_name))

    def re_calculate_image(self):
        value = self.trans_slider.value() / float(self.trans_slider.maximum())
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
