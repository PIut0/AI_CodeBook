"""Datasets
"""

from torch.utils.data import Dataset
import numpy as np
import cv2
import os

class SegDataset(Dataset):
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
        
        self.image_paths = paths
        self.mask_paths = list(map(lambda x : x.replace('images', 'masks').replace('jpg', 'png'),self.image_paths))
        self.input_size = input_size
        self.scaler = scaler
        self.logger = logger
        self.verbose = verbose
        self.mode = mode

        # train-time transform
        self.transform = A.Compose([
            A.ElasticTransform(p = 0.9, alpha = 50, sigma = 12, alpha_affine = 0.2),
            A.HorizontalFlip(p = 0.5), 
            A.VertialFlip(p = 0.5),
        ])
        
        self.normalize = A.Compose([
            A.ToTensor(num_classes = 2, normalize = True)
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, id_: int):
        
        filename = os.path.basename(self.image_paths[id_]) # Get filename for logging
        x = cv2.imread(self.image_paths[id_], cv2.IMREAD_COLOR)
        orig_size = x.shape

        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = cv2.resize(x, (self.input_size, self.input_size))
        x = self.scaler(x)
        x = np.transpose(x, (2, 0, 1))
        
        # ADDED
        x = self.normalize(x)

        if self.mode in ['train', 'valid']:
            y = cv2.imread(self.mask_paths[id_], cv2.IMREAD_GRAYSCALE)
            y = cv2.resize(y, (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)
            
            # ADDED
            transform_image = self.transform(image = x, mask = y)
            x = transform_image['image']
            y = transform_image['mask']
                
            return x, y, filename

        elif self.mode in ['test']:
            return x, orig_size, filename

        else:
            assert False, f"Invalid mode : {self.mode}"


