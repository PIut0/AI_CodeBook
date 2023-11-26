import random
import pandas as pd
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import librosa
from glob import glob
import datasets #HF
from datasets import Dataset, DatasetDict

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import warnings
warnings.filterwarnings(action='ignore')

CFG = {
    'SR':16000,
    'N_MFCC':32, # Melspectrogram 벡터를 추출할 개수
    'SEED':42
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


seed_everything(CFG['SEED']) # Seed 고정f_data=hf_data.train_test_split(train_size=0.8,seed=0)

data_url = "../data"

train_df = pd.read_csv(os.path.join(data_url, 'train.csv'))
test_df = pd.read_csv(os.path.join(data_url, 'test.csv'))

train_df['path'] = data_url + os.sep + train_df['path']
test_df['path'] = data_url + os.sep + test_df['path']

train_data = Dataset.from_pandas(train_df)
test_data = Dataset.from_pandas(test_df)

train_data = train_data.train_test_split(train_size=0.8, seed=CFG['SEED'])
train_data

labels = np.sort(train_df['label'].unique())
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[int(label)] = str(i)  # Convert to Python int
    id2label[str(i)] = int(label)  # Convert to Python int
    
train_data = train_data.cast_column("path", datasets.Audio(sampling_rate=CFG['SR']))
train_data['train'][0]

from transformers import AutoFeatureExtractor
model='facebook/wav2vec2-large-xlsr-53'
feature_extractor = AutoFeatureExtractor.from_pretrained(model)

def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["path"]]
    inputs = feature_extractor(
        audio_arrays, sampling_rate=feature_extractor.sampling_rate, max_length=16000*2, truncation=True
    )
    return inputs

encoded_dataset = train_data.map(preprocess_function, remove_columns=["path"], batched=True)
encoded_dataset

from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer

model='facebook/wav2vec2-large-xlsr-53'
num_labels = len(id2label)
model = AutoModelForAudioClassification.from_pretrained(model, num_labels=num_labels, label2id=label2id, id2label=id2label)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    fp16=False,
    learning_rate=5e-5,
    num_train_epochs=50,
    logging_steps =10,
    per_device_train_batch_size =32,
    per_device_eval_batch_size =32,
    save_total_limit =1,
    push_to_hub=False
)

def compute_metrics(eval_preds):
    metric = datasets.load_metric("accuracy")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset['train'],
    eval_dataset=encoded_dataset["test"],
    tokenizer=feature_extractor,
    compute_metrics =compute_metrics,
)