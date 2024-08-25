import numpy as np
from layer import ILayer
from loss import Loss
from matplotlib import pyplot as plt
import pickle
from learning_rate import LearningRateSchedule, LearningRateScheduler
import signal
import sys

class BrainBuilder:
    
    def __init__(self, layers: list[ILayer], learning_rate: np.float32, loss_type: Loss.LossType, visualization=False, lr_schedule_type=LearningRateSchedule.STEP_DECAY, decay_rate=0.1, decay_steps=1000) -> None:
        self._layers = layers
        self._loss_type = loss_type
        self._loss = Loss(loss_type)
        self._lr = learning_rate
        self._viz = visualization
        self._lr_schedule_type = lr_schedule_type
        self._lr_scheduler = LearningRateScheduler(lr_schedule_type, learning_rate, decay_rate, decay_steps)

        self.stop_training = False
        self.graph_losses = []
        self.losses = []
       
        if self._viz:
            self.fig, self.ax = plt.subplots()
            self.line, = self.ax.plot([], [], lw=2, color='blue')
            self.ax.set_title("Training Loss Over Iterations")
            self.ax.set_xlabel("Iterations")
            self.ax.set_ylabel("Loss")
            self.ax.grid(True)
            self.text_box = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes, fontsize=12,
                                         verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
            self.ax.set_ylim(0, 1)
            self.fig.canvas.mpl_connect('close_event', self.on_close)

        signal.signal(signal.SIGINT, self.signal_handler)

    def feedforward(self, xs: np.ndarray):
        layer_input = np.array(xs).reshape(-1, 1)
        for layer in self._layers:
            layer_input = layer.feedforward(layer_input)
        return layer_input

    def get_loss(self, predicted, output):
        return self._loss.get_loss(predicted, output)

    def learn(self, predicted: np.ndarray, output: np.ndarray):
        dn = self._loss.get_loss_derivative(predicted, output)
        xi = dn
        for layer in self._layers[::-1]:
            xi = layer.learn(self._lr_scheduler.get_lr(), xi)

    def train(self, max_iter, datas, labels):
        self.stop_training = False
        if self._viz:
            plt.ion()
            plt.show()

        for i in range(max_iter):
            if self.stop_training:
                print("Training stopped.")
                break
            self.losses = []
            for j in range(len(datas)):
                if self.stop_training:
                    print("Training stopped.")
                    break
                
                prediction = self.feedforward(datas[j])
                loss = self.get_loss(prediction, labels[j])
                self.losses.append(loss)
                self.learn(prediction, labels[j])
            
            self._lr_scheduler.update()

            if self._viz:
                self.update_plot(i, np.average(self.losses))
            if i % 100 == 0 or i == max_iter - 1:
                print(f"({i+1}/{max_iter}) iteration, Loss: {loss:.6f}")

    def update_plot(self, iteration, loss):
        self.graph_losses.append(loss)
        self.line.set_data(range(len(self.graph_losses)), self.graph_losses)
        self.ax.relim()
        self.ax.autoscale_view()
        self.text_box.set_text(f"Iteration: {iteration}\nMin Loss: {min(self.graph_losses):.4f}\nLearining Rate({self._lr_schedule_type.name}): {self._lr_scheduler.get_lr()}")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def predict(self, datas, labels):
        for i, data in enumerate(datas):
            res = self.feedforward(data)
            print(labels[i].T, " => ", res.T)

    def on_close(self, event):
        """Handle the matplotlib close event."""
        print("Matplotlib window closed.")
        self.stop_training = True

    def signal_handler(self, sig, frame):
        """Handle keyboard interrupts."""
        print("Keyboard interrupt received.")
        self.stop_training = True

    def save(self, filename: str):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename: str):
        with open(filename, 'rb') as file:
            return pickle.load(file)
