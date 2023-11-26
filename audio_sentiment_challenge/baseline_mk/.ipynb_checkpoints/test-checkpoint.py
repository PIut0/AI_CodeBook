import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple

from datetime import datetime
import audiomentations
import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transformers
from datasets import Dataset
from prettytable import PrettyTable
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, MulticlassRecall
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from transformers.feature_extraction_utils import BatchFeature
from transformers import AutoFeatureExtractor, WhisperForAudioClassification, AutoTokenizer, AutoModelForSpeechSeq2Seq
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from optimum.bettertransformer import BetterTransformer

transformers.logging.set_verbosity_error()

# feature_extractor = AutoFeatureExtractor.from_pretrained("sanchit-gandhi/whisper-medium-fleurs-lang-id")
# model = WhisperForAudioClassification.from_pretrained("sanchit-gandhi/whisper-medium-fleurs-lang-id")
# tokenizer = AutoTokenizer.from_pretrained("sanchit-gandhi/whisper-medium-fleurs-lang-id")
# ds = load_dataset("google/fleurs", "all", split="validation", streaming=True)

# from transformers import TrainingArguments
# import evaluate

# training_args = TrainingArguments(output_dir = "test_trainer")
# metric = evaluate.load("accuracy")

# feature_extractor = AutoFeatureExtractor.from_pretrained('openai/whisper-base')

processor = AutoProcessor.from_pretrained("openai/whisper-base")
model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-base")
