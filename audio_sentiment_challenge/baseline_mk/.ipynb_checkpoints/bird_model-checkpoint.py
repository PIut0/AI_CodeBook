#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import gc
import ast
import random
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
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
from torch.optim import Adam, AdamW
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


class config:
    seed = 42
    num_fold = 1
    sample_rate = 16000
    n_fft = 1024
    hop_length = 512
    n_mels = 64
    duration = 5
    num_classes = 6
    train_batch_size = 128
    valid_batch_size = 128
    model_name = 'efficientnet_v2_l'
    epochs = 20
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    learning_rate = 1e-7


# In[ ]:


# def seed_everything(seed):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
# seed_everything(config.seed)


# In[ ]:


df = pd.read_csv("/scratch/network/mk8574/audio_sentiment_challenge/data/train.csv")
df.head()


# In[ ]:


signal, sr = torchaudio.load('/scratch/network/mk8574/audio_sentiment_challenge/data/train/TRAIN_0001.wav')
print(signal.shape)
print(sr)


# In[ ]:


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

        signal, sr = torchaudio.load(audio_path) # loaded the audio
        
        # Now we first checked if the sample rate is same as TARGET_SAMPLE_RATE and if it not equal we perform resampling
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        
        # IN CASE DATA IS STEREO:
        # Next we check the number of channels of the signal
        #signal -> (num_channels, num_samples) - Eg.-(2, 14000) -> (1, 14000)
        if signal.shape[0]>1:
            signal = torch.mean(signal, axis=0, keepdim=True)
        

        # Lastly we check the number of samples of the signal
        #signal -> (num_channels, num_samples) - Eg.-(1, 14000) -> (1, self.num_samples)
        # If it is more than the required number of samples, we truncate the signal
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        
        # If it is less than the required number of samples, we pad the signal
        if signal.shape[1]<self.num_samples:
            num_missing_samples = self.num_samples - signal.shape[1]
            last_dim_padding = (0, num_missing_samples)
            signal = F.pad(signal, last_dim_padding)
        
        # Finally all the process has been done and now we will extract mel spectrogram from the signal
        mel = self.transformation(signal)
        
        # For pretrained models, we need 3 channel image, so for that we concatenate the extracted mel
        image = torch.cat([mel, mel, mel])
        
        # Normalize the image
        max_val = torch.abs(image).max()
        image = image / max_val
        
        label = torch.tensor(self.labels[index])
        
        if self.mode in ['train', 'valid']:
            return image, label
        
        else:
            return image


# In[ ]:


from sklearn.model_selection import train_test_split

mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=config.sample_rate, 
                                                      n_fft=config.n_fft, 
                                                      hop_length=config.hop_length, 
                                                      n_mels=config.n_mels)
# Function to get data according to the folds
def get_data():
    df = pd.read_csv('/scratch/network/mk8574/audio_sentiment_challenge/data/train.csv')
    train_df, valid_df = train_test_split(df, test_size = 0.2, shuffle = True)
    
    train_dataset = AudioSentDataset(train_df, mel_spectrogram, config.sample_rate, config.duration, mode = 'train')
    valid_dataset = AudioSentDataset(valid_df, mel_spectrogram, config.sample_rate, config.duration, mode = 'valid')
    
    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.valid_batch_size, shuffle=True)
    
    return train_loader, valid_loader


# In[ ]:


class BirdCLEFResnet(nn.Module):
    def __init__(self):
        super(BirdCLEFResnet, self).__init__()
        self.base_model = models.__getattribute__(config.model_name)(pretrained=True)
        # for param in self.base_model.parameters():
        #     param.requires_grad = False

        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p = 0.4, inplace = True),
            nn.Linear(in_features = 1280, out_features = 6, bias = True)
        )
        
        torch.nn.init.xavier_normal_(self.base_model.classifier[1].weight.data)

#         in_features = self.base_model.fc.in_features
        
#         self.base_model.fc = nn.Linear(in_features, config.num_classes)
#         torch.nn.init.xavier_normal_(self.base_model.fc.weight.data)

#         (classifier): Sequential(
    #   (0): Dropout(p=0.4, inplace=True)
    #   (1): Linear(in_features=1280, out_features=1000, bias=True)
    # )

    def forward(self, x):
        x = self.base_model(x)
        return x


# In[ ]:


BirdCLEFResnet()


# In[ ]:


from torch import autograd
autograd.set_detect_anomaly(True)


# In[108]:


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
        
        # print(mels.shape)
        # print('-' * 60)
        # print(labels.shape)
        
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


# In[109]:


def valid(model, data_loader, device, epoch):
    model.eval()
    
    running_loss = 0
    pred = []
    label = []
    
    loop = tqdm(data_loader, position=0)
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


# In[110]:


a = torch.Tensor([1, 2, 3])
b = torch.Tensor([1, 4, 5])

print((a == b).float().sum())


# In[113]:


def run():
    train_loader, valid_loader = get_data()
    
    model = BirdCLEFResnet().to(config.device)
    
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=1e-5, T_max=10)
    
    best_valid_f1 = 0
    for epoch in range(config.epochs):
        train_loss = train(model, train_loader, optimizer, scheduler, config.device, epoch)
        valid_loss, valid_f1, valid_acc = valid(model, valid_loader, config.device, epoch)
        
        print(f"Validation F1 - {valid_f1}, Accuracy - {valid_acc}")
        torch.save(model.state_dict(), f'./model_{epoch}.bin')
        print(f"Saved model checkpoint at ./model_{epoch}.bin")

    return best_valid_f1


# In[ ]:


run()

