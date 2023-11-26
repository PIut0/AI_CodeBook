from config import Config
from modules.trainer import train
from modules.model import MyModel
from modules.dataset import MyDataSet
from modules.dataloader import collate_fn_yes_label, collate_fn_no_label
from modules.recorder import Recorder

import os
import json
import torch
import random
import warnings
import numpy as np
import pandas as pd
import audiomentations

from datetime import datetime
from dataclasses import asdict
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

warnings.filterwarnings(action='ignore')

train_serial = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = f"/scratch/network/mk8574/audio_sentiment_challenge/baseline_youhan/results/{train_serial}"

config = Config()

device = torch.device(f'cuda:{config.device}') if torch.cuda.is_available() else torch.device('cpu')
print(f'train at :{device}')

if config.logging:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, "config.json"), "w") as config_file:
        json.dump(asdict(config), config_file, indent=4, sort_keys=False)

    print(f'config file saved at: {train_serial}')

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(config.seed)

train_df = pd.read_csv(os.path.join(config.data_path, 'train.csv'))
train_df, valid_df = train_test_split(train_df, test_size=config.val_size, random_state=config.seed)

train_df.reset_index(drop=True, inplace=True)
valid_df.reset_index(drop=True, inplace=True)

feature_extractor = AutoFeatureExtractor.from_pretrained(config.pretrained_name)

train_transforms = audiomentations.OneOf(
[
    audiomentations.AddGaussianNoise(p=0.75),
    audiomentations.PitchShift(p=0.75),
    audiomentations.PeakingFilter(p=0.75),
    audiomentations.SevenBandParametricEQ(p=0.75),
    audiomentations.BandPassFilter(p=0.75),
    audiomentations.BandStopFilter(p=0.75),
    audiomentations.AirAbsorption(p=0.75),
    audiomentations.ClippingDistortion(p=0.75),
    audiomentations.HighPassFilter(p=0.75),
    audiomentations.HighShelfFilter(p=0.75),
    audiomentations.Limiter(p=0.75),
    audiomentations.LowPassFilter(p=0.75),
    audiomentations.LowShelfFilter(p=0.75),
])

train_dataset = MyDataSet(train_df, feature_extractor, mode='train', transforms=train_transforms, data_path=config.data_path)
valid_dataset = MyDataSet(valid_df, feature_extractor, mode='valid', data_path=config.data_path)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn_yes_label, num_workers=config.num_workers)
valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn_yes_label, num_workers=config.num_workers)

model = MyModel(config.pretrained_name)

creterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.AdamW(
    [{"params": module.parameters(), "lr": config.lr if name == "classifier" else config.lr * 0.1} for name, module in model.named_children()],
    weight_decay=0.1,
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
recorder = Recorder(save_dir) if config.logging else None

best_model = train(model, creterion, train_loader, valid_loader, optimizer, scheduler, recorder, config, amp=config.amp)

if config.logging:
    best_model.save_pretrained(save_dir)

print('End train, model saved')