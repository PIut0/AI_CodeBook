"""Datasets
"""

from torch.utils.data import Dataset
import numpy as np
import cv2
import os


class ImgDataSet(Dataset):
    """Dataset for image segmentation

    Attributs:
        x_dirs(list): 이미지 경로
        y_dirs(list): 
        input_size(list, tuple): 이미지 크기(width, height)
        scaler(obj): 이미지 스케일러 함수
        logger(obj): 로거 객체
        verbose(bool): 세부 로깅 여부
    """   
    def __init__(self, ids, input_size, scaler, mode='train', logger=None, verbose=False):
        
        self.ids = ids
        self.input_size = input_size
        self.scaler = scaler
        self.logger = logger
        self.verbose = verbose
        self.mode = mode
        self.train_csv = pd.read_csv('scratch/network/mk8574/audio_sentiment_challenge/data/train.csv')

    def __len__(self):
        return len(self.ids) # input range(5001)
    
    def plot_mfcc(wave_file):
        # Load the audio file
        y, sr = sf.read(wave_file)

        # Compute the MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20) # 20 channels
        
        return mfccs # numpy array

    def __getitem__(self, index: int):
        ID = f'TRAIN_{index}.wav'
        
        PATH = f'/scratch/network/mk8574/audio_sentiment_challenge/data/train/TRAIN_{index}.wav'
        
#        filename = os.path.basename(self.x_paths[id_]) # Get filename for logging

        x = sf.read(PATH)

#         x = cv2.imread(self.x_paths[id_], cv2.IMREAD_COLOR)
#         orig_size = x.shape

#         x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
#         x = cv2.resize(x, self.input_size)
#         x = self.scaler(x)
#         x = np.transpose(x, (2, 0, 1))

        if self.mode in ['train', 'valid']:
            y = self.train_csv[self.train_csv['id'] == ID]['label']
            
            return x, y

        elif self.mode in ['test']:
            return x, orig_size

        else:
            assert False, f"Invalid mode : {self.mode}"
