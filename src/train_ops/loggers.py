import torch
import torchvision

__all__ = ["Logger"]

class Logger():
    def __init__(self, log_every, writer, name="train"):
        self.log_every = log_every
        self.writer = writer
        self.name = name

    def log(self, train_iter, metrics, images):
        if train_iter.val % self.log_every == 0:
            for key, value in metrics.items():
                try:
                    self.writer.add_scalar(f'{self.name}/{key}', value.mean(), train_iter.val)
                except:
                    self.writer.add_scalar(f'{self.name}/{key}', value, train_iter.val)
            for key, value in images.items():
                # only logs first element in batch
                img = torchvision.utils.make_grid(value)
                self.writer.add_image(f'{self.name}/{key}', img.clamp(0, 1), train_iter.val)
