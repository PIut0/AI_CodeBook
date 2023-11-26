"""Datasets
"""
#변경됨
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
import albumentations as A




class BBoxDataset(Dataset):
    """Dataset for image segmentation

    Attributs:
        x_dirs(list): 이미지 경로
        y_dirs(list): 마스크 이미지 경로
        input_size(list, tuple): 이미지 크기(width, height)
        scaler(obj): 이미지 스케일러 함수
        logger(obj): 로거 객체
        verbose(bool): 세부 로깅 여부
    """   
    def __init__(self, paths, input_size, scaler, mode='train', logger=None, verbose=False):
        
        self.x_paths = paths
        self.y_paths = list(map(lambda x : x.replace('x', 'y'),self.x_paths))
        self.input_size = input_size
        self.scaler = scaler
        self.logger = logger
        self.verbose = verbose
        self.mode = mode
        self.transform =  A.Compose([
            A.Crop(x_min=0, y_min=0, x_max=1024, y_max=1024, always_apply=True)
            #A.VerticalFlip(p=0.5),
            #A.ShiftScaleRotate(shift_limit=0.0625, p=0.5),
            #A.Blur(p=0.1)
            
        ])

    def __len__(self):
        return len(self.x_paths)

    def __getitem__(self, id_: int):
        
        filename = os.path.basename(self.x_paths[id_]) # Get filename for logging
        x = cv2.imread(self.x_paths[id_], cv2.IMREAD_COLOR)
        orig_size = x.shape
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = cv2.resize(x, self.input_size)
        
        
       

        if self.mode in ['train', 'valid']:
            y = cv2.imread(self.y_paths[id_], cv2.IMREAD_GRAYSCALE)
            y = cv2.resize(y, self.input_size, interpolation=cv2.INTER_NEAREST)
            transformed = self.transform(image=x,mask=y)
            x = transformed['image'] #albumenations
            y = transformed['mask']
            x = self.scaler(x)
            x = np.transpose(x, (2, 0, 1))
            return x, y, filename

        elif self.mode in ['test']:
            x = self.scaler(x)
            x = np.transpose(x, (2, 0, 1))
            return x, orig_size, filename

        else:
            assert False, f"Invalid mode : {self.mode}"
            
            
            
            



