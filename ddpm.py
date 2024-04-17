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
# from torchview import draw_graph
# from digit_classifier import *
# import logging
# from torch.utils.tensorboard import SummaryWriter

# logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

#inference code = "https://github.com/dome272/Diffusion-Models-pytorch/blob/main/ddpm_conditional.py"

myseed = 6666
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

train_tfm = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize(40),
            transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device) # beta is a list of several point 
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = device

    def prepare_noise_schedule(self): # beta schedule
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t): # training used
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n): # training used
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, labels, cfg_scale=3):
        # logging.info(f"Sampling {n} new images....")
        print(f"Sampling {n} new images...")
        model.eval()

        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            # x_0 = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            # x_1 = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            # x_2 = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            # x_3 = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            # x_4 = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            # x_5 = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x) # randn_like = normal distribution
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                # if (i == 1):
                #     x_0 = x
                #     x_0 = (x_0.clamp(-1, 1) + 1) / 2
                #     x_0 = (x_0 * 255).type(torch.uint8)
                # if (i == 99):
                #     x_1 = x
                #     x_1 = (x_1.clamp(-1, 1) + 1) / 2
                #     x_1 = (x_1 * 255).type(torch.uint8)
                # if (i == 199):
                #     x_2 = x
                #     x_2 = (x_2.clamp(-1, 1) + 1) / 2
                #     x_2 = (x_2 * 255).type(torch.uint8)
                # if (i == 299):
                #     x_3 = x
                #     x_3 = (x_3.clamp(-1, 1) + 1) / 2
                #     x_3 = (x_3 * 255).type(torch.uint8)
                # if (i == 499):
                #     x_4 = x
                #     x_4 = (x_4.clamp(-1, 1) + 1) / 2
                #     x_4 = (x_4 * 255).type(torch.uint8)
                # if (i == 599):
                #     x_5 = x
                #     x_5 = (x_5.clamp(-1, 1) + 1) / 2
                #     x_5 = (x_5 * 255).type(torch.uint8)
                
        model.train()
        # print(type(x))
        # print(x[0].size())
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        # print(x.size())
        # print(type(x))

        # print(x_0.size())
        # print(type(x_0))
        # return x, x_0, x_1, x_2, x_3, x_4, x_5
        return x

def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def train():
    # setup_logging(args.run_name)

    # setting up parameters 
    epochs_n = 200
    batch_size = 32
    image_size = 32
    num_classes = 10
    run_name = "DDPM_conditional"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = "./hw2_data/digits/mnistm/"
    lr = 3e-4

    train_set = mnistDataset(root=root, tfm=train_tfm, status="train")
    trainLoader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = UNet_conditional(num_classes=num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=image_size, device=device)
    # logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(trainLoader)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    OUTPUT_FILE_PATH = "./result/" 
    if not os.path.exists(OUTPUT_FILE_PATH):
        os.makedirs(OUTPUT_FILE_PATH)
    for epoch in range(epochs_n):
        # logging.info(f"Starting epoch {epoch}:")
        train_loss = []
        print(f"Training epoch {epoch+1}\n")
        # pbar = tqdm(trainloader)
        for images, labels in tqdm(trainLoader):
            images = images.to(device)
            labels = labels.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            if np.random.random() < 0.1:
                labels = None
            predicted_noise = model(x_t, t, labels)
            loss = mse(noise, predicted_noise)
            # print(type(model_graph))
            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)

            # pbar.set_postfix(MSE=loss.item())
            # logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        train_loss = sum(train_loss) / len(train_loss)
        print(f"[ Train | {epoch + 1:03d}/{epochs_n:03d} ] loss = {train_loss:.5f}")
        if epoch % 10 == 0:
            labels = torch.arange(10).long().to(device)
            sampled_images = diffusion.sample(model, n=len(labels), labels=labels)
            # ema_sampled_images = diffusion.sample(ema_model, n=len(labels), labels=labels)
            # plot_images(sampled_images)
            # save_images(sampled_images, os.path.join(OUTPUT_FILE_PATH, f"{epoch}.jpg"))
            # save_images(ema_sampled_images, os.path.join("results", run_name, f"{epoch}_ema.jpg"))
            # torch.save(model.state_dict(), f"{epoch+1}ckpt.pt")
            # torch.save(ema_model.state_dict(), os.path.join("models", run_name, f"ema_ckpt.pt"))
            # torch.save(optimizer.state_dict(), os.path.join("models", run_name, f"optim.pt"))

            # calculate accuracy
            # OUTPUT_FILE_PATH = os.path.join("./results", run_name)
            # print(f"{epoch+1} acc:")
            # calculate_acc(OUTPUT_FILE_PATH)
            # print("\n")

            # clean up the results file
            # for f in os.listdir(OUTPUT_FILE_PATH):
            #     os.remove(os.path.join(OUTPUT_FILE_PATH, f))

def main():
    train()


if __name__ == '__main__':
    main()
    # device = "cuda"
    # model = UNet_conditional(num_classes=10).to(device)
    # ckpt = torch.load("./models/DDPM_conditional/ckpt.pt")
    # model.load_state_dict(ckpt)
    # diffusion = Diffusion(img_size=64, device=device)
    # n = 8
    # y = torch.Tensor([6] * n).long().to(device)
    # x = diffusion.sample(model, n, y, cfg_scale=0)
    # plot_images(x)
