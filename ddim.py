import os
import copy
import sys
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from PIL import Image
# from utils import *
from UNet import *  
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,  Dataset
import torchvision.utils
import torch.nn.functional as F

#reference code = "https://github.com/dome272/Diffusion-Models-pytorch/blob/main/ddpm_conditional.py"
#reference = "https://zhuanlan.zhihu.com/p/565698027"

myseed = 6666
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)


class DDIM:
    def __init__(self, noise_steps=1000, ddim_step=50,beta_start=1e-4, beta_end=2e-2, img_size=256, device="cuda"):
        self.device = device
        self.noise_steps = noise_steps
        self.ddim_step = ddim_step
        self.ddimstep_sch = self.ddimstep_schedule()
        self.ddimstep_pre_sch = self.ddimstep_pre_schedule()
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device) # beta is a list of several point 
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size

    def prepare_noise_schedule(self): # beta schedule
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps, dtype=torch.float64)

    def ddimstep_schedule(self):
        return torch.linspace(0, 980, self.ddim_step, dtype=torch.float64) # return [0, 20, 40, ... 980] 50 步
    
    def ddimstep_pre_schedule(self):
        zero_tensor = torch.tensor([0])
        return torch.cat((zero_tensor,self.ddimstep_sch), dim=0)
    
    def sigma(self, eta, alpha, alpha_pre):
        return eta * torch.sqrt((1 - alpha_pre) / (1 - alpha) * (1 - alpha / alpha_pre))

    def sample(self, model, x, n, cfg_scale=3, eta=0):
        print(f"Sampling {n} new images...")
        model.eval()

        with torch.no_grad():
            # x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device) # load noise!!!
            x = x.float().to(self.device)
            for i in tqdm(reversed(range(1, self.ddim_step)), position=0): # 50, 49, 48, 47
                t = (torch.ones(n) * self.ddimstep_sch[i]).float().to(self.device)
                prev_t = (torch.ones(n) * self.ddimstep_pre_sch[i]).float().to(self.device)
                predicted_noise = model(x, t)
                # print(x)
                alpha = self.alpha_hat[t.long()][:, None, None, None]
                alpha_pre = self.alpha_hat[prev_t.long()][:, None, None, None] 
                alpha_hat = self.alpha_hat[t.long()][:, None, None, None] 
                sigma_ = self.sigma(eta, alpha, alpha_pre) 
                if i > 1:
                    noise = torch.randn_like(x) # randn_like = normal distribution
                else:
                    noise = torch.zeros_like(x)
                x = torch.sqrt(alpha_pre) * ((x - torch.sqrt(1 - alpha) * predicted_noise) / torch.sqrt(alpha))
                x += torch.sqrt(1 - alpha_pre - torch.square(sigma_)) * predicted_noise + sigma_ * noise

                x = x.to(torch.float32)
                   
        model.train()
        x = x.clamp(-1,1)

        return x

def loadNoise(noise_path): # should return [10, 3, 256, 256]

    filelist = []
    filenameList = []
    for filename in sorted(os.listdir(noise_path)):
        # print(filename)
        filelist.append(torch.load(os.path.join(noise_path, filename)))
        filenameList.append(filename.split(".")[0])
    allNoise = torch.cat(filelist, dim=0)

    return allNoise, filenameList

def MSE(ground_truth_path, generated_path):

    trans = transforms.ToTensor()
    for img in sorted(os.listdir(ground_truth_path)):
        gt_img = Image.open(os.path.join(ground_truth_path, img))
        gr_img = Image.open(os.path.join(generated_path, img))

        gt_img = trans(gt_img)
        gr_img = trans(gr_img)

        mse = F.mse_loss(gt_img, gr_img)

        print(f"{img}_mse_loss: ","{:.8f}".format(mse.item()))

# draw grid for report
def report_grid(path, model):
    report_noise = loadNoise(path)

    report_img_generated = 4
    eta_0 = ddim.sample(model, report_noise, img_generated, eta=0)
    eta_0_25 = ddim.sample(model, report_noise, img_generated, eta=0.25)
    eta_0_5 = ddim.sample(model, report_noise, img_generated, eta=0.5)
    eta_0_75 = ddim.sample(model, report_noise, img_generated, eta=0.75)
    eta_1 = ddim.sample(model, report_noise, img_generated, eta=1)

    report_image = torch.cat((eta_0, eta_0_25, eta_0_5 , eta_0_75, eta_1), dim=0)

    grid = torchvision.utils.make_grid(report_image, nrow=4)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save("./p2_report.png")

def slerp(img1, img2, alpha):

    theta = torch.acos(torch.sum(img1 * img2) / (torch.norm(img1) * torch.norm(img2)))

    result = torch.sin((1 - alpha) * theta) / torch.sin(theta) * img1 + torch.sin(alpha * theta) / torch.sin(theta) * img2
    return result

def linear(img1, img2, alpha):
    return (1 - alpha) * img1 + (alpha) * img2

def interpolation(noise, device, model, ddim):

    img1 = noise[0].unsqueeze(0)
    img2 = noise[1].unsqueeze(0)

    result_list = []
    for alpha in range(11):
        alpha = alpha / 10.0
        # result = slerp(img1, img2, alpha)
        result = linear(img1, img2, alpha)
        result_list.append(result)
    
    all_result = torch.cat(result_list, dim=0)
    # print(all_result.size())
    report_img_generated = 11
    image = ddim.sample(model, all_result, report_img_generated, eta=0) # sampling

    # print(image.size())
    grid = torchvision.utils.make_grid(image, nrow=11)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save("./p2_linear_interpolation.png")


def main(NOISE_PATH, OUTPUT_FILE_PATH, MODEL_PATH):

    # load model type
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet().to(device)
    ddim = DDIM(device=device)
    model.load_state_dict(torch.load(MODEL_PATH))

    # load noise into file
    all_noise, filenameList = loadNoise(NOISE_PATH)

    # sample
    img_generated = all_noise.shape[0]
    sampled_image = ddim.sample(model, all_noise, img_generated)

    # # output to a directory
    # GROUND_TRUTH_PATH = './hw2_data/face/GT/'

    if not os.path.exists(OUTPUT_FILE_PATH):
        os.makedirs(OUTPUT_FILE_PATH)

    for idx in range(sampled_image.shape[0]):
        img = sampled_image[idx]
        imgPath = filenameList[idx] + ".png"
        # print(imgPath)
        path = os.path.join(OUTPUT_FILE_PATH, imgPath)
        # print(path)
        torchvision.utils.save_image(img, path, normalize=True)
        
    

    # calculate MSE and print MSE
    # MSE(GROUND_TRUTH_PATH, OUTPUT_FILE_PATH)

    # DRAW_NOISE_PATH = "./face_p2_report/"
    # noise = loadNoise(DRAW_NOISE_PATH)
    # print(noise.size())

    # interpolation(noise, device, model, ddim)

if __name__ == '__main__':

    NOISE_PATH = sys.argv[1]
    OUTPUT_FILE_PATH = sys.argv[2]
    MODEL_PATH = sys.argv[3]

    # NOISE_PATH = "./hw2_data/face/noise"
    # OUTPUT_FILE_PATH = "./p2_result_multi"
    # MODEL_PATH = "./hw2_data/face/UNet.pt"

    main(NOISE_PATH, OUTPUT_FILE_PATH, MODEL_PATH)
