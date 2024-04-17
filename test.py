import os
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from PIL import Image
# from utils import *
from conditionU import UNet_conditional, EMA
from p1_dataloader import mnistDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,  Dataset
import torchvision.utils
# from digit_classifier import *
from ddpm import Diffusion
import sys

myseed = 6666
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

def save_images(images, path, label,**kwargs):
    for idx in range(images.shape[0]):
        img = images[idx]
        img = transforms.Resize(28,antialias=True)(img)
        img = Image.fromarray(img.permute(1,2,0).cpu().numpy())
        imgPath = str(label) + "_" + str('{0:03d}'.format(idx+1)) + ".png"
        outpath = os.path.join(path, imgPath)
        # outpath = path + str(label) + "_" + str('{0:03d}'.format(idx+1)) + ".png"
        # print(outpath)
        img.save(outpath)

def continuous_image(images, path,**kwargs):
    img = images[0]
    img = transforms.Resize(28,antialias=True)(img)
    img = Image.fromarray(img.permute(1,2,0).cpu().numpy())
    img.save(path)

def draw_for_report():
    n_img_generated=10 
    all_labels = torch.full(size=(1,n_img_generated), fill_value=0).long().to(device)
    all_labels = all_labels.squeeze(0)
    for i in range(1, 10):
        labels = torch.full(size=(1,n_img_generated), fill_value=i).long().to(device)
        labels = labels.squeeze(0)
        all_labels = torch.cat((all_labels, labels), 0)
    sampled_images, one, two, three, four, five, six = diffusion.sample(model, n=100, labels=all_labels)
    grid = torchvision.utils.make_grid(sampled_images, nrow=10)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    # print(ndarr.size())
    im = Image.fromarray(ndarr)
    im.save("./p1_report.png")

    continuous_image(one, "./0.png")
    continuous_image(two, "./100.png")
    continuous_image(three, "./200.png")
    continuous_image(four, "./300.png")
    continuous_image(five, "./500.png")
    continuous_image(six, "./600.png")


def main(OUTPUT_FILE_PATH): 

    MODEL_PATH = "./p1_model.pt" # load MODEL
    image_size = 32
    num_classes = 10
    noise_steps = 400

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet_conditional(num_classes=num_classes).to(device)
    diffusion = Diffusion(noise_steps=noise_steps, img_size=image_size, device=device)
    model.load_state_dict(torch.load(MODEL_PATH), strict=False)

    # sampling picture to directory
    # create directory
    # OUTPUT_FILE_PATH = "./test/" 
    if not os.path.exists(OUTPUT_FILE_PATH):
        os.makedirs(OUTPUT_FILE_PATH)
    
    # # sampling pictures 10 for each
    # labels = torch.arange(10).long().to(device) # labels = [0,1,2,3,4,5,6,7,8,9]
    n_img_generated = 100 # 每次sample 100 張
    for i in range(10):
        labels = torch.full(size=(1,n_img_generated), fill_value=i).long().to(device)
        labels = labels.squeeze(0)
        # print(f"Sampling{i+1} label:")
        sampled_images = diffusion.sample(model, n=n_img_generated, labels=labels)
        save_images(sampled_images, OUTPUT_FILE_PATH, i)
        # print(sampled_images.shape) #[10, 3, 32, 32] 10 pictures, 3*32*32
        # print(type(sampled_images)) # torch.Tensor


    # drawing for report 
    # n_img_generated=10 
    # all_labels = torch.full(size=(1,n_img_generated), fill_value=0).long().to(device)
    # all_labels = all_labels.squeeze(0)
    # for i in range(1, 10):
    #     labels = torch.full(size=(1,n_img_generated), fill_value=i).long().to(device)
    #     labels = labels.squeeze(0)
    #     all_labels = torch.cat((all_labels, labels), 0)
    # sampled_images, one, two, three, four, five, six = diffusion.sample(model, n=100, labels=all_labels)
    # grid = torchvision.utils.make_grid(sampled_images, nrow=10)
    # ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    # # print(ndarr.size())
    # im = Image.fromarray(ndarr)
    # im.save("./p1_report.png")


    # continuous_image(one, "./0.png")
    # continuous_image(two, "./100.png")
    # continuous_image(three, "./200.png")
    # continuous_image(four, "./300.png")
    # continuous_image(five, "./500.png")
    # continuous_image(six, "./600.png")



if __name__ == "__main__":
    OUTPUT_FILE_PATH = sys.argv[1]
    main(OUTPUT_FILE_PATH)
