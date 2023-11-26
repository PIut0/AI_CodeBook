import torch.nn as nn
import audiomentations
from dataclasses import dataclass
from audiomentations.core.transforms_interface import BaseTransform

@dataclass
class Config:
    device: int = 1
    num_workers: int = 10
    logging: bool = True # if set logging=True, model & records will be saved
    
    # data args
    data_path = "/scratch/network/mk8574/audio_sentiment_challenge/data"
    val_size: float = 0.2
    
    # model args
    pretrained_name: str = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    
    # hparams
    seed: int = 42
    max_epoch: int = 50
    lr: float = 5e-4
    batch_size: int = 8
    total_batch_size: int = 32
    early_stop_patience = 5
    amp: bool = True
