from config import Config, Train_transforms, Train_settings
from modules.seed import seed_everything
from modules.trainer import train
from modules.model import MyModel
from modules.dataset import MyDataSet
from modules.dataloader import collate_fn_yes_label, collate_fn_no_label
from modules.recorder import Recorder

import os
import json
import torch
import shutil
import warnings
import numpy as np
import pandas as pd

from dataclasses import asdict
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from transformers import ViTImageProcessor

warnings.filterwarnings(action='ignore')
config = Config()

save_dir = f"/scratch/network/mk8574/audio_sentiment_challenge/baseline_youhan/results/{config.train_serial}"
device = torch.device(f'cuda:{config.device}') if torch.cuda.is_available() else torch.device('cpu')

print(f'train at :{device}')

if config.logging:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    config_path_from = '/scratch/network/mk8574/audio_sentiment_challenge/baseline_youhan/config.py'
    config_path_to = os.path.join(save_dir, 'config.py')
    
    shutil.copyfile(config_path_from, config_path_to)
    
    print(f'config file saved at: {config.train_serial}')

seed_everything(config.seed)

train_df = pd.read_csv(os.path.join(config.data_path, 'train.csv'))
train_df, valid_df = train_test_split(train_df, test_size=config.val_size, random_state=config.seed, stratify=train_df['label'])

train_df.reset_index(drop=True, inplace=True)
valid_df.reset_index(drop=True, inplace=True)

feature_extractor = AutoFeatureExtractor.from_pretrained(config.pretrained_name)

train_transforms = Train_transforms()
train_dataset = MyDataSet(train_df, feature_extractor, mode='train', transforms=train_transforms.transforms, data_path=config.data_path)
valid_dataset = MyDataSet(valid_df, feature_extractor, mode='valid', data_path=config.data_path)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn_yes_label, num_workers=config.num_workers)
valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn_yes_label, num_workers=config.num_workers)

model = MyModel(config.pretrained_name)

train_settings = Train_settings(model, config)

recorder = Recorder(save_dir) if config.logging else None

best_model = train(model, train_settings.creterion, train_loader, valid_loader, train_settings.optimizer, train_settings.scheduler, recorder, config, amp=config.amp, save_dir=config.logging*save_dir)

if config.logging:
    best_model.save_pretrained(save_dir)

print('End train, model saved')