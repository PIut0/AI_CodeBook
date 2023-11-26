"""Datasets
"""
#변경됨
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
import albumentations as A
##
from skimage.color import lab2lch, rgb2lab
from skimage.exposure import rescale_intensity
from skimage.morphology import disk
from sklearn.cluster import KMeans
import gc
import pandas as pd
##


def custom_shadow_augmentation(image,**kwargs):
    # Shadow Detection
    struc_elem_size = 5
    convolve_window_size = 5
    mask_size=5
    num_thresholds=3
    struct_elem_size=5
    exponent=1
    #image = rescale_intensity(np.transpose(f.read(tuple(np.arange(4)), [1, 2, 0]), out_range = 'uint8'))
   
    image = image[:, :, 0 : 3]
        
    lch_img = np.float32(lab2lch(rgb2lab(image)))
    # ... (perform the rest of the shadow detection steps from your shadow_detection function)
  
    l_norm = rescale_intensity(lch_img[:, :, 0], out_range = (0, 1))
    h_norm = rescale_intensity(lch_img[:, :, 2], out_range = (0, 1))
    sr_img = (h_norm + 1) / (l_norm + 1)
    log_sr_img = np.log(sr_img + 1)
    
    del l_norm, h_norm, sr_img
    gc.collect()

    

    avg_kernel = np.ones((convolve_window_size, convolve_window_size)) / (convolve_window_size ** 2)
    blurred_sr_img = cv2.filter2D(log_sr_img, ddepth = -1, kernel = avg_kernel)
      
    
    del log_sr_img
    gc.collect()
    
    flattened_sr_img = blurred_sr_img.flatten().reshape((-1, 1))
    labels = KMeans(n_clusters = num_thresholds + 1, max_iter = 10000, n_init=10).fit(flattened_sr_img).labels_
    flattened_sr_img = flattened_sr_img.flatten()
    df = pd.DataFrame({'sample_pixels': flattened_sr_img, 'cluster': labels})
    threshold_value = df.groupby(['cluster']).min().max().iloc[0]
    df['Segmented'] = np.uint8(df['sample_pixels'] >= threshold_value)
    
    
    del blurred_sr_img, flattened_sr_img, labels, threshold_value
    gc.collect()
    
    
    shadow_mask_initial = np.array(df['Segmented']).reshape((image.shape[0], image.shape[1]))
    struc_elem = disk(struc_elem_size)
    shadow_mask = np.expand_dims(np.uint8(cv2.morphologyEx(shadow_mask_initial, cv2.MORPH_CLOSE, struc_elem)), axis = 0)
    
    
    del df, shadow_mask_initial, struc_elem
    gc.collect()
    
    # Shadow Correction
    # ... (perform shadow correction steps from your shadow_correction function)
    corrected_img = np.zeros((image.shape), dtype = np.uint8)
    non_shadow_mask = np.uint8(shadow_mask == 0)
    
    
    for i in range(image.shape[2]):
        shadow_area_mask = shadow_mask * image[:, :, i]
        non_shadow_area_mask = non_shadow_mask * image[:, :, i]
        shadow_stats = np.float32(np.mean(((shadow_area_mask ** exponent) / np.sum(shadow_mask))) ** (1 / exponent))
        non_shadow_stats = np.float32(np.mean(((non_shadow_area_mask ** exponent) / np.sum(non_shadow_mask))) ** (1 / exponent))
        mul_ratio = ((non_shadow_stats - shadow_stats) / shadow_stats) + 1
        corrected_img[:, :, i] = np.uint8(non_shadow_area_mask + np.clip(shadow_area_mask * mul_ratio, 0, 255))
    

    

    
    return corrected_img  # Return the corrected image


##
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
        
        self.x_paths = paths
        self.y_paths = list(map(lambda x : x.replace('x', 'y'),self.x_paths))
        self.input_size = input_size
        self.scaler = scaler
        self.logger = logger
        self.verbose = verbose
        self.mode = mode
        self.transform =  A.Compose([
            A.VerticalFlip(p=0.5),
            A.Blur(p=0.2)
            #A.Lambda(image=custom_shadow_augmentation,p=1)
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
            
            
            
            



