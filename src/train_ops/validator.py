
import torch

from tqdm import tqdm

from src.train_ops.loggers import Logger
from src.loss.losses import non_saturating_loss, calc_mse, calc_psnr, loss_func

import sys

__all__ = ["Validator"]

class Validator():
    def __init__(self, writer, dataloader, args):
        self.dataloader = dataloader
        self.logger = Logger(1, writer, "test")
        self.run_every = int(args.valid_every)

    def eval(self, gen, lpips, train_iter, args):
        if train_iter.val % self.run_every == 0:
            with tqdm(total=len(self.dataloader), desc="Validate", position=-1) as t:
                for i, (x, y) in enumerate(self.dataloader):

                    gen.eval()

                    with torch.no_grad():
                        x, y = x.to(args.device), y.to(args.device)
                        x_pred = gen(y)
                        mse_loss = calc_mse(x_pred, x)
                        lpips_loss = lpips.forward(x_pred, x, normalize=True)

                    if i == 0:
                        metrics = {'mse': mse_loss, "lpips": lpips_loss}
                        images = {f'high_res_{i}': x, f"low_res_{i}": y, f"super_res_{i}": x_pred}
                    else:
                        tmp_metrics = {'mse': mse_loss, "lpips": lpips_loss}
                        tmp_images = {f'high_res_{i}': x, f"low_res_{i}": y, f"super_res_{i}": x_pred}
                        metrics = self._append(metrics, tmp_metrics)
                        images = {**images, **tmp_images}
                    t.update()
            metrics = self._agg(metrics)
            self.logger.log(train_iter, metrics, images)

    def _append(self, metrics, tmp_metrics):
        for k, _ in metrics.items():
            metrics[f"{k}"] += tmp_metrics[f"{k}"]
        return metrics

    def _agg(self, metrics):
        for k, _ in metrics.items():
            metrics[f"{k}"] /= len(self.dataloader)
        return metrics
