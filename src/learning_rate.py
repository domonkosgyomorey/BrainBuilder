import numpy as np
from enum import Enum

class LearningRateSchedule(Enum):
    STEP_DECAY = 1
    EXPONENTIAL_DECAY = 2
    TIME_BASED_DECAY = 3
    COSINE_ANNEALING = 4

class LearningRateScheduler:
    def __init__(self, schedule_type: LearningRateSchedule, initial_lr: np.float32, decay_rate: float = 0.1, decay_steps: int = 1000, total_steps: int = 10000) -> None:
        self.schedule_type = schedule_type
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.total_steps = total_steps
        self.iteration = 0
        self.current_lr = initial_lr

    def get_lr(self):
        return self.current_lr

    def update(self):
        if self.schedule_type == LearningRateSchedule.STEP_DECAY:
            self.current_lr = self.initial_lr * (self.decay_rate ** (self.iteration // self.decay_steps))
        
        elif self.schedule_type == LearningRateSchedule.EXPONENTIAL_DECAY:
            self.current_lr = self.initial_lr * np.exp(-self.decay_rate * self.iteration)
        
        elif self.schedule_type == LearningRateSchedule.TIME_BASED_DECAY:
            self.current_lr = self.initial_lr / (1 + self.decay_rate * self.iteration)
        
        elif self.schedule_type == LearningRateSchedule.COSINE_ANNEALING:
            self.current_lr = self.initial_lr * (1 + np.cos(np.pi * self.iteration / self.total_steps)) / 2
        
        self.iteration += 1
