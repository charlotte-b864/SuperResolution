import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Function
from torch.utils.data import DataLoader

import argparse
from tqdm import tqdm
from PIL import Image

from src.network.models import Generator
from src.data.dataloader import ImageDataset

parser = argparse.ArgumentParser(description="Super-Resolution")

parser.add_argument("--gen_ch", type=float, default=384)
parser.add_argument("--load_dir", type=str, default=None)
parser.add_argument("--input_image", type=str, default=None)
parser.add_argument("--output_image", type=str, default=None)
parser.add_argument("--cuda_id", type=str, default="cpu")
parser.add_argument("--bicubic_output", type=str, default=None)

args = parser.parse_args()
if args.cuda_id == "cpu":
    args.device = "cpu"
else:
    args.device = torch.device("".join(["cuda:", f"{args.cuda_id}"]) if torch.cuda.is_available() else "cpu")

to_tensor = lambda x: transforms.ToTensor()(x)[None]
to_PIL = lambda x: transforms.ToPILImage()(x.detach().cpu().squeeze(0))

def infer(args):
    gen = Generator(args.gen_ch).to(args.device)

    try:
        print("Loading model...")
        checkpoint = torch.load(args.load_dir, map_location=args.device)
        gen.load_state_dict(checkpoint["gen"])
        print("Model load complete.")
    except:
        print('Model files misspecified or does not exist.')

    gen.eval()

    y = Image.open(args.input_image)
    y_torch = to_tensor(y).to(args.device)

    with torch.no_grad():
        x = gen(y_torch).clamp(0, 1)
    x = to_PIL(x)

    if args.bicubic_output is not None:
        y_baseline = y.resize([int(4*d) for d in y.size], Image.BICUBIC)
        y_baseline.save(args.bicubic_output)

    x.save(args.output_image)

if __name__ == "__main__":
    infer(args)
