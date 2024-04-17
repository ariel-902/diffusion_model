import pandas as pd
import torch
import os
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from PIL import Image
# from torchvision.datasets import DatasetFolder, VisionDataset

class mnistDataset(Dataset):
    def __init__(self,root,tfm, status="train"): # where root is hw2_data/digits/mnistm/
        super(mnistDataset).__init__()
        self.root = root
        self.filesname = []
        self.transform=tfm
        self.csvFile = pd.read_csv(os.path.join(root, f'{status}.csv'))
        # print(csvFile)
        # print(len(csvFile.index))
        # self.length = len(csvFile.index)

        imgPath = root+"/data/"
        for idx in range(len(self.csvFile.index)):
            self.filesname.append([os.path.join(imgPath, self.csvFile["image_name"][idx]), self.csvFile["label"][idx]])  #[imgPath, label]
    
    
    def __getitem__(self, idx):
        imgName = self.filesname[idx][0]
        label = self.filesname[idx][1]
        img = Image.open(imgName)
        if (self.transform != None):
            img = self.transform(img)
            # print(img.type())

        return img, label
    
    def __len__(self):
        return len(self.csvFile.index)

    