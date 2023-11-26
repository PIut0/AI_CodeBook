import os
import librosa
import numpy as np
from torch.utils.data import Dataset

# import sys
# sys.path.append('/scratch/network/mk8574/audio_sentiment_challenge/EXPORT/')
# from augmentations import *
# from EDA import *


class MyDataSet(Dataset):
    def __init__(self, df, feature_extractor, mode='train', transforms=None, data_path=None):
        self.df = df
        self.feature_extractor = feature_extractor 
        self.mode = mode
        self.transforms = transforms
        self.data_path = data_path
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = os.path.join(self.data_path, self.df['path'][idx][2:])
        
        waveform, sample_rate = librosa.load(path)
        sr = self.feature_extractor.sampling_rate
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=sr)
        
        if self.transforms != None:
            waveform = self.transforms(samples=np.array(waveform, dtype=np.float32), sample_rate=sr)
        
        input_values = self.feature_extractor(waveform, sampling_rate=sr, return_tensors="pt").input_features[0]#변경
    
        #input_values = self.feature_extractor(waveform, sampling_rate=sr, return_tensors="pt", padding=True).input_values
        if self.mode != 'test':
            label = self.df['label'][idx]
            return input_values.squeeze(), label
        else:
            return input_values.squeeze()
