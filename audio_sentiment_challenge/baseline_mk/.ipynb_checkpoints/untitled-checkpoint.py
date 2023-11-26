import os
import json
import random
import warnings
from datetime import datetime
from dataclasses import asdict, dataclass
from sklearn.model_selection import train_test_split

import librosa
import audiomentations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

warnings.filterwarnings(action='ignore')


@dataclass
class Config:
    device: int = 0
    # data args
    data_path: str = "/scratch/network/mk8574/audio_sentiment_challenge/data"
    val_size: float = 0.2
    
    train_transforms: audiomentations.core.composition.OneOf = audiomentations.OneOf([
        audiomentations.AddGaussianNoise(p = 0.4),
        audiomentations.Clip(p = 0.5),
        audiomentations.Shift(p = 0.7)
    ])
    
    # # [
    #     audiomentations.AddGaussianNoise(p=0.75),
    #     audiomentations.PitchShift(p=0.75),
    #     audiomentations.PeakingFilter(p=0.75),
    #     audiomentations.SevenBandParametricEQ(p=0.75),
    #     audiomentations.BandPassFilter(p=0.75),
    #     audiomentations.BandStopFilter(p=0.75),
    #     audiomentations.AirAbsorption(p=0.75),
    #     audiomentations.ClippingDistortion(p=0.75),
    #     audiomentations.HighPassFilter(p=0.75),
    #     audiomentations.HighShelfFilter(p=0.75),
    #     audiomentations.Limiter(p=0.75),
    #     audiomentations.LowPassFilter(p=0.75),
    #     audiomentations.LowShelfFilter(p=0.75),
    # ]) = None

    # save dir
    train_serial = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir: str = f"/scratch/network/mk8574/audio_sentiment_challenge/baseline_mk/results/{train_serial}"

    # model args
    pretrained_name: str = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"

    # hparams
    seed: int = 42
    max_epoch: int = 32
    lr: float = 5e-4
    batch_size: int = 32
    total_batch_size: int = 32
    gradient_accumulate_step: int = 1  # total batch size = batch_size * gradient_accumulate_step
    early_stop_patience = 5
    
config = Config()

device = torch.device(f'cuda:{config.device}') if torch.cuda.is_available() else torch.device('cpu')
print(device)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(config.seed) # SEEDING


train_df = pd.read_csv(os.path.join(config.data_path, 'train.csv'))
train_df, valid_df = train_test_split(train_df, test_size=config.val_size, random_state=config.seed)

train_df.reset_index(drop=True, inplace=True)
valid_df.reset_index(drop=True, inplace=True)

class MyDataSet(Dataset):
    def __init__(self, df, feature_extractor, mode='train', transforms=None):
        self.df = df
        self.feature_extractor = feature_extractor
        self.mode = mode
        self.transforms = transforms
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = os.path.join(config.data_path, self.df['path'][idx][2:])
        
        waveform, sample_rate = librosa.load(path)
        sr = self.feature_extractor.sampling_rate
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=sr)
        
        if self.transforms is not None:
            waveform = self.transforms(samples=np.array(waveform, dtype=np.float32), sample_rate=sr)
        
        input_values = self.feature_extractor(waveform, sampling_rate=sr, return_tensors="pt", padding=True).input_values
        
        if self.mode!= 'test':
            label = self.df['label'][idx]
            return input_values.squeeze(), label
        else:
            return input_values.squeeze()

        
def collate_fn_yes_label(batch):
    waveforms, labels = zip(*batch)
    waveforms = pad_sequence([torch.tensor(wave) for wave in waveforms], batch_first=True)
    labels = torch.tensor(labels)
    return waveforms, labels

def collate_fn_no_label(batch):
    waveforms = zip(*batch)
    waveforms = pad_sequence([torch.tensor(wave) for wave in waveforms], batch_first=True)
    return waveforms

# print(type(audiomentations.OneOf([audiomentations.AddGaussianNoise(p = 0.4),audiomentations.Clip(p = 0.4)])))
# print(config.train_transforms)

feature_extractor = AutoFeatureExtractor.from_pretrained(config.pretrained_name)

train_dataset = MyDataSet(train_df, feature_extractor, mode='train', transforms=config.train_transforms)
valid_dataset = MyDataSet(valid_df, feature_extractor, mode='val')

train_loader = DataLoader(train_dataset,
                          batch_size=config.batch_size,
                          shuffle=True,
                          collate_fn=collate_fn_yes_label,
                          num_workers=16)
valid_loader = DataLoader(valid_dataset,
                          batch_size=config.batch_size,
                          shuffle=True,
                          collate_fn=collate_fn_yes_label,
                          num_workers=16)


class MyModel(torch.nn.Module):
    def __init__(self, pretrained_name: str):
        super(MyModel, self).__init__()
        self.model = AutoModelForAudioClassification.from_pretrained(pretrained_name)
        
        self.model.classifier = nn.Linear(in_features=self.model.projector.out_features, out_features=6)
        nn.init.kaiming_normal_(self.model.classifier.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.model.classifier.bias)

    def forward(self, x):
        output = self.model(x)
        return output.logits
    
def validation(model, valid_loader, creterion):
    model.eval()
    val_loss = []

    total, correct = 0, 0

    with torch.no_grad():
        for waveforms, labels in tqdm(iter(valid_loader)):
            waveforms = waveforms.to(device)
            labels = labels.to(device)

            output = model(waveforms)            
            loss = creterion(output, labels)

            val_loss.append(loss.item())

            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += predicted.eq(labels).cpu().sum()

    accuracy = correct / total

    avg_loss = np.mean(val_loss)

    return avg_loss, accuracy



def train(model, train_loader, valid_loader, optimizer, scheduler):
    accumulation_step = int(config.total_batch_size / config.batch_size)
    model.to(device)
    creterion = nn.CrossEntropyLoss(label_smoothing = 0.0).to(device)
    best_model = None
    best_acc = 0

    for epoch in range(1, config.max_epoch+1):
        train_loss = []
        model.train()
        
        for i, (waveforms, labels) in enumerate(tqdm(train_loader)):
            waveforms = waveforms.to(device)
            labels = labels.flatten().to(device)

            optimizer.zero_grad()
            
            output = model(waveforms)
            loss = creterion(output, labels)
            loss.backward()

            if (i+1) % accumulation_step == 0:
                optimizer.step()
                optimizer.zero_grad()

            train_loss.append(loss.item())

        avg_loss = np.mean(train_loss)
        valid_loss, valid_acc = validation(model, valid_loader, creterion)

        if scheduler is not None:
            scheduler.step(valid_loss)

        if valid_acc > best_acc:
            best_acc = valid_acc
            best_model = model

        print(f'epoch:[{epoch}] train loss:[{avg_loss:.5f}] valid_loss:[{valid_loss:.5f}] valid_acc:[{valid_acc:.5f}]')
    
    print(f'best_acc:{best_acc:.5f}')

    return best_model

model = MyModel(config.pretrained_name)

optimizer = torch.optim.AdamW(
    [{"params": module.parameters(), "lr": config.lr if name == "classifier" else config.lr * 0.1} for name, module in model.named_children()],
    weight_decay=0.1,
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

best_model = train(model, train_loader, valid_loader, optimizer, scheduler)