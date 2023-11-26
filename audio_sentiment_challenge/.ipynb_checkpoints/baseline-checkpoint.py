import random
import pandas as pd
import numpy as np
import os
from tqdm.auto import tqdm
import librosa

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

seed_everything(CFG['SEED']) # Seed 고정

train_df = pd.read_csv('/scratch/network/mk8574/audio_sentiment_challenge/data/train.csv')
test_df = pd.read_csv('/scratch/network/mk8574/audio_sentiment_challenge/data/test.csv')

def get_mfcc_feature(df):
    features = []
    for path in tqdm(df['path']):
        # librosa패키지를 사용하여 wav 파일 load
        y, sr = librosa.load(path, sr=CFG['SR'])
        # librosa패키지를 사용하여 mfcc 추출
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=CFG['N_MFCC'])
        y_feature = []
        # 추출된 MFCC들의 평균을 Feature로 사용
        for e in mfcc:
            y_feature.append(np.mean(e))
        features.append(y_feature)

    mfcc_df = pd.DataFrame(features, columns=['mfcc_'+str(x) for x in range(1,CFG['N_MFCC']+1)])
    return mfcc_df

train_x = get_mfcc_feature(train_df)
test_x = get_mfcc_feature(test_df)

train_y = train_df['label']

model = DecisionTreeClassifier(random_state=CFG['SEED'])
model.fit(train_x, train_y)

preds = model.predict(test_x)

submission = pd.read_csv('/scratch/network/mk8574/audio_sentiment_challenge/data/sample_submission.csv')
submission['label'] = preds
submission.to_csv('/scratch/network/mk8574/audio_sentiment_challenge/data/baseline_submission.csv', index=False)