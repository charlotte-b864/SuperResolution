import os
from PIL import Image
from random import randrange
from tqdm import tqdm


train_root = "./train/"
valid_root = "./valid/"

def random_crop(img, dims=(256, 256)):
    h, w = img.size
    h_crop = randrange(0, h - dims[0])
    w_crop = randrange(0, w - dims[1])
    return img.crop((h_crop, w_crop, h_crop + dims[0], w_crop + dims[1]))


def proc_imgs(f_dir, num_samples, scale=4, crop=None):
    f_raw = f_dir + "raw/"
    f_hr = f_dir + "high_res/"
    f_lr = f_dir + "low_res/"
    f_names = [img for img in os.listdir(f_raw) if img.split(".")[-1] == "png"]
    count = 0
    with tqdm(total=num_samples, desc=f"Processing {f_dir}") as t:
        while count < num_samples:
            for j, f_name in enumerate(f_names):
                img = Image.open(f"{f_raw}{f_name}")
                if crop is not None:
                    hr_img = random_crop(img, dims=crop)
                    crop_to = crop
                else:
                    hr_img = img
                    crop_to = img.size
                lr_img = hr_img.resize([d//scale for d in crop_to], Image.BICUBIC)
                hr_img.save(f"{f_hr}{count}.png")
                lr_img.save(f"{f_lr}{count}.png")
                count += 1
                t.update()

if __name__ == "__main__":
    proc_imgs(valid_root, 100, 4, crop=None)
    proc_imgs(train_root, 25000, 4, [256, 256])
