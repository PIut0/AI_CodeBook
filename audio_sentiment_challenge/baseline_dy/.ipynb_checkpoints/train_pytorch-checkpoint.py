import os
import gc
import ast
import random
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns
from tqdm import tqdm
import torchaudio
import IPython.display as ipd
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models

import warnings
warnings.filterwarnings('ignore')


#######################



import augly.audio as audaugs
import augly.utils as utils
from augly.audio.utils import validate_and_load_audio

########################3
aug = audaugs.Compose([
    audaugs.AddBackgroundNoise(p = 0.1),
    audaugs.Clip(duration_factor = 0.7),
#    audaugs.TimeStretch(rate = 3.0),
#    audaugs.Speed(factor = 3.0),
    audaugs.Harmonic(p = 0.5),
    audaugs.InvertChannels(),
    audaugs.OneOf([audaugs.Clicks(p = 0.6),
                   audaugs.InsertInBackground(offset_factor = 0.25, p = 0.6)
                   ])
#    audaugs.ToMono()
])


#######################
class config:
    seed = 42
    num_fold = 1
    sample_rate = 16000
    n_fft = 1024
    hop_length = 512
    n_mels = 64
    duration = 5
    num_classes = 6
    train_batch_size = 64
    valid_batch_size = 32
    model_name = 'swin_v2_s'
    epochs = 20
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    learning_rate = 1e-4
    
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(config.seed)


df = pd.read_csv("/scratch/network/mk8574/audio_sentiment_challenge/data/train.csv")
# aug = audaugs.Compose([
#     audaugs.AddBackgroundNoise(p = 0.1),
#     audaugs.Clip(duration_factor = 0.7),
# #    audaugs.TimeStretch(rate = 3.0),
# #    audaugs.Speed(factor = 3.0),
#     audaugs.Harmonic(p = 0.5),
#     #audaugs.InvertChannels(),
#     audaugs.OneOf([audaugs.Clicks(p = 0.6),
#                    audaugs.InsertInBackground(offset_factor = 0.25, p = 0.6)
#                    ])
# #    audaugs.ToMono()
# ])
# aug = audaugs.Compose([
#     audaugs.AddBackgroundNoise(p = 0.1),
#     audaugs.Clip(duration_factor = 0.7),
# #    audaugs.TimeStretch(rate = 3.0),
# #    audaugs.Speed(factor = 3.0),
#     audaugs.Harmonic(p = 0.5),
#     audaugs.InvertChannels(),
#     audaugs.OneOf([audaugs.Clicks(p = 0.6),
#                    audaugs.InsertInBackground(offset_factor = 0.25, p = 0.6)
#                    ])
# #    audaugs.ToMono()
# ])


class AudioSentDataset(Dataset):
    def __init__(self, df, transformation, target_sample_rate, duration, mode):
        self.audio_paths = df['path'].values
        self.labels = df['label'].values
        self.transformation = transformation # transformation
        self.target_sample_rate = target_sample_rate # sample rate
        self.num_samples = target_sample_rate * duration
        self.mode = mode # ['train', 'valid', 'test']
        
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, index):
        audio_path = os.path.join('/scratch/network/mk8574/audio_sentiment_challenge/data', self.audio_paths[index])

        #signal, sr = torchaudio.load(audio_path) # loaded the audio
        signal, sr = validate_and_load_audio(audio_path)
        signal, _ = aug(signal, sample_rate = sr, metadata = [])
        signal = torch.Tensor(signal)
        
        # Now we first checked if the sample rate is same as TARGET_SAMPLE_RATE and if it not equal we perform resampling
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        
        # IN CASE DATA IS STEREO:
        # Next we check the number of channels of the signal
        #signal -> (num_channels, num_samples) - Eg.-(2, 14000) -> (1, 14000)
#         if signal.shape[0]>1:
#             signal = torch.mean(signal, axis=0, keepdim=True)
        

        # Lastly we check the number of samples of the signal
        #signal -> (num_channels, num_samples) - Eg.-(1, 14000) -> (1, self.num_samples)
        # If it is more than the required number of samples, we truncate the signal
        if signal.shape[0] > self.num_samples:
            signal = signal[:self.num_samples]
        
        # If it is less than the required number of samples, we pad the signal
        if signal.shape[0]<self.num_samples:
            num_missing_samples = self.num_samples - signal.shape[0]
            last_dim_padding = (0, num_missing_samples)
            signal = F.pad(signal, last_dim_padding)
        signal = signal.unsqueeze(1).transpose(0,1)
        
        # Finally all the process has been done and now we will extract mel spectrogram from the signal
        mel = self.transformation(signal)
        
        # For pretrained models, we need 3 channel image, so for that we concatenate the extracted mel
        image = torch.cat([mel, mel, mel])
        
        # Normalize the image
        max_val = torch.abs(image).max()
        #print(max_val)
        image = image / max_val
       
        label = torch.tensor(self.labels[index])
        
        if self.mode in ['train', 'valid']:
            return image, label
        
        else:
            return image
        
        
from sklearn.model_selection import train_test_split

mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=config.sample_rate, 
                                                      n_fft=config.n_fft, 
                                                      hop_length=config.hop_length, 
                                                      n_mels=config.n_mels)



mfcc = torchaudio.transforms.MFCC(sample_rate = config.sample_rate,
                                n_mfcc = 20,
                                log_mels = False)

# Function to get data according to the folds
def get_data():
    df = pd.read_csv('/scratch/network/mk8574/audio_sentiment_challenge/data/train.csv')
    train_df, valid_df = train_test_split(df, test_size = 0.2, shuffle = True)
    
    train_dataset = AudioSentDataset(train_df, mfcc, config.sample_rate, config.duration, mode = 'train')
    valid_dataset = AudioSentDataset(valid_df, mfcc, config.sample_rate, config.duration, mode = 'valid')
    
    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.valid_batch_size, shuffle=True)
    
    return train_loader, valid_loader

class BirdCLEFResnet(nn.Module):
    def __init__(self):
        super(BirdCLEFResnet, self).__init__()
        self.base_model = models.__getattribute__(config.model_name)(pretrained=True)
        #for param in self.base_model.parameters():
            #param.requires_grad = False
            
        #in_features = self.base_model.head.out_features
        
        #self.base_model.head.out_features = nn.Linear(in_features, config.num_classes)
        self.base_model.head.out_features=6
    def forward(self, x):
        x = self.base_model(x)
        return x
    
def loss_fn(outputs, labels):
    SMOOTH = 1e-10
    
    return nn.CrossEntropyLoss(label_smoothing = 0.3)(outputs + SMOOTH, labels)

def train(model, data_loader, optimizer, scheduler, device, epoch):
    model.train()
    
    running_loss = 0
    loop = tqdm(data_loader, position=0)
    for i, (mels, labels) in enumerate(loop):
        mels = mels.to(device)
        labels = labels.to(device)
        
        outputs = model(mels)
        _, preds = torch.max(outputs, 1)
        
        
        
        loss = loss_fn(outputs, labels)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        
        optimizer.step()
        optimizer.zero_grad()
        
        if scheduler is not None:
            scheduler.step()
        running_loss += loss.item()
        
        loop.set_description(f"Epoch [{epoch+1}/{config.epochs}]")
        loop.set_postfix(loss=loss.item())

    return running_loss / len(data_loader)


def valid(model, data_loader, device, epoch):
    model.eval()
    
    running_loss = 0
    pred = []
    label = []
    
    loop = tqdm(data_loader, position=1)
    for mels, labels in loop:
        mels = mels.to(device)
        labels = labels.to(device)
        
        outputs = model(mels)
        _, preds = torch.max(outputs, 1)
        
        loss = loss_fn(outputs, labels)
        # print('Outputs:', outputs)
        # print('Labels:', labels)
            
        running_loss += loss.item()
        
        pred.extend(preds.view(-1).cpu().detach().numpy())
        label.extend(labels.view(-1).cpu().detach().numpy())
        
        loop.set_description(f"Epoch [{epoch+1}/{config.epochs}]")
        loop.set_postfix(loss=loss.item())
        
    valid_f1 = f1_score(label, pred, average='macro')
    label = torch.Tensor(label)
    pred = torch.Tensor(pred)
    valid_acc = (label == pred).float().sum() / label.shape[0]
    
    return running_loss/len(data_loader), valid_f1, valid_acc
import datetime
def run():
    now = datetime.datetime.now()
    nowDatetime = now.strftime('%Y%m%d%H%M%S')
    train_loader, valid_loader = get_data()
    
    model = BirdCLEFResnet().to(config.device)
    
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=1e-5, T_max=10)
    
    best_valid_f1 = 0
    for epoch in tqdm(range(config.epochs),position=0):
        train_loss = train(model, train_loader, optimizer, scheduler, config.device, epoch)
        valid_loss, valid_f1, valid_acc = valid(model, valid_loader, config.device, epoch)
        
        print(train_loss,valid_loss)
        if valid_f1 > best_valid_f1:
            print(f"Validation Accuracy Improved - {best_valid_f1} ---> {valid_f1}")
            torch.save(model.state_dict(), f'./model_{nowDatetime}.bin')
            print(f"Saved model checkpoint at ./model_{nowDatetime+"-"+str(epoch)}.bin")
            best_valid_f1 = valid_f1
        else:
            print(f"Validation Score : BEST: {best_valid_f1} NOW: {valid_f1}")

    return best_valid_f1

from torch import autograd
autograd.set_detect_anomaly(True)
run()