import torch
import torch.nn as nn
import pandas as pd
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class QualDataset(Dataset):
    def __init__(self, dataframe, preprocessor, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.preprocessor = preprocessor
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['img_path']
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        # mos column 존재 여부에 따라 값을 설정
        mos = float(self.dataframe.iloc[idx]['mos']) if 'mos' in self.dataframe.columns else 0.0
        comment = self.dataframe.iloc[idx]['comments'] if 'comments' in self.dataframe.columns else ""
        
        encoding = self.preprocessor(
            images = img, 
            text = comment, 
            max_length = 16,
            padding = "longest", 
            return_tensors = "pt",
            pad_to_multiple_of = 8, 
            #truncation = True,
        )
        
        #return img, mos, comment
        return img, comment