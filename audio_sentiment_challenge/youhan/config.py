# mixup 겹치기 epoch 늘려서

import torch
import torch.nn as nn
import audiomentations
from datetime import datetime
from dataclasses import dataclass
from audiomentations.core.transforms_interface import BaseTransform

@dataclass
class Config:
    device: int = 0
    num_workers: int = 10
    logging: bool = True # if set logging=True, model & records will be saved 
    train_serial: int = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # data args
    data_path = "/scratch/network/mk8574/audio_sentiment_challenge/data"
    val_size: float = 0.0
    
    # model args
    pretrained_name: str = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    
    # hparams
    seed: int = 42
    max_epoch: int = 30
    lr: float = 5e-4
    batch_size: int = 8
    total_batch_size: int = 50
    early_stop_patience = 5
    amp: bool = True
    
    mixup: bool = False
    mixup_idx: int = 3
    alpha: float = 1.5

class Train_transforms:
    def __init__(self): 
        self.transforms = audiomentations.OneOf([
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
        
class Train_settings:
    def __init__(self, model, config):
        self.creterion = nn.CrossEntropyLoss().to(config.device)
        self.optimizer = torch.optim.AdamW(
            [{"params": module.parameters(), "lr": config.lr if name == "classifier" else config.lr * 0.1} for name, module in model.named_children()],
            weight_decay=0.1,
        )
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[13,25], gamma=0.5)