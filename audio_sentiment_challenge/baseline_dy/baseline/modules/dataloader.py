import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn_yes_label(batch):
    waveforms, labels = zip(*batch)
    waveforms = pad_sequence([wave.clone().detach() for wave in waveforms], batch_first=True)
    labels = torch.tensor(labels)
    return waveforms, labels

def collate_fn_no_label(batch):
    waveforms = pad_sequence([wave.clone().detach() for wave in batch], batch_first=True)
    return waveforms