import torch.nn as nn
import audiomentations
from dataclasses import dataclass
from audiomentations.core.transforms_interface import BaseTransform
from datetime import datetime
from dataclasses import dataclass
@dataclass
class Config:
    device: int = 0
    num_workers: int = 10
    logging: bool = True # if set logging=True, model & records will be saved 
    train_serial: int = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # data args
    data_path = "/scratch/network/mk8574/audio_sentiment_challenge/data"
    val_size: float = 0.2
    
    # model args
    #pretrained_name: str = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    #pretrained_name : str = "shhossain/whisper-tiny-bn-emo" 
    pretrained_name : str = "/scratch/network/mk8574/audio_sentiment_challenge/baseline_dy/results/whisper_mid"
    # hparams
    seed: int = 42
    max_epoch: int = 50
    lr: float = 5e-4
    batch_size: int = 4
    total_batch_size: int = 4
    early_stop_patience = 5
    amp: bool = True
    mixup: bool = False
    mixup_idx: int = 3
    alpha: float = 2
