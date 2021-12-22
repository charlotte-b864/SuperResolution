import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Function
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import argparse
from tqdm import tqdm

from src.loss.losses import non_saturating_loss, calc_mse, calc_psnr, loss_func
from src.loss.perceptual_similarity import perceptual_loss as ps
from src.network.models import Discriminator, Generator
from src.data.dataloader import ImageDataset
from src.ops.other import init_sys, TrainIter, ModelSaver
from src.train_ops.loggers import Logger
from src.train_ops.validator import Validator

parser = argparse.ArgumentParser(description="Super-Resolution")

parser.add_argument("--batch_size", type=float, default=8)
parser.add_argument("--warm_start", type=float, default=25000)
parser.add_argument("--valid_every", type=float, default=5000)
parser.add_argument("--save_every", type=float, default=5000)
parser.add_argument("--log_every", type=float, default=25)
parser.add_argument("--epochs", type=float, default=100)
parser.add_argument("--gen_ch", type=float, default=384)
parser.add_argument("--learn_rate", type=float, default=1e-4)
parser.add_argument("--adv_coef", type=float, default=0.15)
parser.add_argument("--mse_coef", type=float, default=0.075*2**(-5))
parser.add_argument("--lpips_coef", type=float, default=1.0)
parser.add_argument("--lr_schedule", type=list, default=[150000, 200000])

parser.add_argument("--dataset", type=str, default="/home/tom/Documents/charlotte_super_res/data/")
parser.add_argument("--exp_name", type=str, default="_test_run_")
parser.add_argument("--save_dir", type=str, default="/home/tom/Documents/charlotte_super_res/model_saves/")
parser.add_argument("--tensorboard_dir", type=str, default="/home/tom/Documents/charlotte_super_res/logging/")
parser.add_argument("--load_dir", type=str, default=None)
parser.add_argument("--cuda_id", type=str, default="0")
parser.add_argument("--denoise", type=bool, default=False)

args = parser.parse_args()
args.device = torch.device("".join(["cuda:", f"{args.cuda_id}"]) if torch.cuda.is_available() else "cpu")

writer = init_sys(args)

def train(args):

    train_set = ImageDataset(root_dir=f"{args.dataset}train/", denoise=args.denoise, transform=transforms.ToTensor())
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    test_set = ImageDataset(root_dir=f"{args.dataset}valid/", denoise=args.denoise, transform=transforms.ToTensor())
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    gen = Generator(args.gen_ch).to(args.device)
    disc = Discriminator().to(args.device)

    lpips = ps.PerceptualLoss(model="net-lin", net="alex", use_gpu=True, gpu_ids=[args.device.index])

    opt_gen = optim.Adam(gen.parameters(), lr=args.learn_rate)
    opt_disc = optim.Adam(disc.parameters(), lr=args.learn_rate)

    train_iter = TrainIter()
    validator = Validator(writer, test_loader, args)
    model_saver = ModelSaver(args)
    train_logger = Logger(args.log_every, writer)

    if args.load_dir is not None:
        print("Loading model...")
        train_iter.load(args.load_dir)
        checkpoint = torch.load(args.load_dir, map_location=args.device)
        gen.load_state_dict(checkpoint["gen"])
        disc.load_state_dict(checkpoint["disc"])
        opt_gen.load_state_dict(checkpoint["opt_gen"])
        opt_disc.load_state_dict(checkpoint["opt_disc"])
        print("Model load complete.")

    scheduler_gen = optim.lr_scheduler.MultiStepLR(opt_gen, [i - train_iter.val for i in args.lr_schedule], gamma=0.1)
    scheduler_disc = optim.lr_scheduler.MultiStepLR(opt_disc, [i - train_iter.val for i in args.lr_schedule], gamma=0.1)

    for e in range(args.epochs):
        with tqdm(total=len(train_loader), desc="Train", position=-1) as t:
            for x, y, in train_loader:
                gen.train()
                disc.train()

                x, y = x.to(args.device), y.to(args.device)
                x_pred = gen(y)

                if train_iter.val > args.warm_start:
                    opt_disc.zero_grad()

                    real_out, real_out_logits = disc(x, y)
                    fake_out, fake_out_logits = disc(x_pred.detach(), y)
                    disc_loss = non_saturating_loss(real_out_logits, fake_out_logits, disc=True)

                    disc_loss.backward(retain_graph=True)
                    nn.utils.clip_grad_norm_(disc.parameters(), 2.50)
                    opt_disc.step()
                else:
                    real_out, fake_out, disc_loss, adv_loss = [torch.zeros(1).to(args.device)]*4

                opt_gen.zero_grad()

                mse_loss = calc_mse(x_pred, x)
                lpips_loss = lpips.forward(x_pred, x, normalize=True)

                if train_iter.val > args.warm_start:
                    fake_out, fake_out_logits = disc(x_pred, y)
                    adv_loss = non_saturating_loss(real_out_logits, fake_out_logits, disc=False)

                loss = loss_func(lpips_loss, mse_loss, adv_loss, args)

                loss.mean().backward()
                nn.utils.clip_grad_norm_(gen.parameters(), 2.50)
                opt_gen.step()

                metrics = {'mse': mse_loss, "lpips": lpips_loss, "loss": loss, "adv_loss": adv_loss, "disc_loss": disc_loss, "real_out": real_out, "fake_out": fake_out}
                images = {'high_res': x, "low_res": y, "super_res": x_pred}

                train_logger.log(train_iter, metrics, images)
                validator.eval(gen, lpips, train_iter, args)

                model_saver.save({"gen": gen.state_dict(), "disc": disc.state_dict(), "opt_gen": opt_gen.state_dict(), "opt_disc": opt_disc.state_dict()}, train_iter)

                scheduler_gen.step()
                scheduler_disc.step()

                train_iter.increment()
                t.update()

if __name__ == "__main__":
    train(args)
