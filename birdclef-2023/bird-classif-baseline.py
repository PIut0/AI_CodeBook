#!/usr/bin/env python
# coding: utf-8

# In[46]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from sklearn import model_selection
import torchvision.transforms as transforms
import torchvision.io 
import librosa
from PIL import Image
import albumentations as A
import torch.multiprocessing as mp
import warnings

import librosa as lb
import librosa.display as lbd
import soundfile as sf
from  soundfile import SoundFile
from  IPython.display import Audio
from pathlib import Path
from tqdm.notebook import tqdm
import joblib, json, re
from pytorch_lightning.callbacks import ModelCheckpoint, BackboneFinetuning, EarlyStopping
from  sklearn.model_selection  import StratifiedKFold
tqdm.pandas()


warnings.filterwarnings('ignore')


# In[44]:


class Config:
    use_aug = False
    num_classes = 264 # output classes
    batch_size = 64 # batch size for train, test
    epochs = 50 
    PRECISION = 16 # precision
    PATIENCE = 8 # early stopping patience (can turn on/off patience later)
    seed = 2023
    model = 'tf_efficientnet_b7_ns' # model: resnet-based OR dense OR efficientnet
    pretrained = True # getting from timm package
    weight_decay = 1e-3
    use_mixup = True
    mixup_alpha = 0.2
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # data-specific directories
    data_root = '.'
    train_images = './train'
    valid_images = './valid'
    
    train_path = './specs/train'
    valid_path = './specs/valid'
    test_path = '/scratch/network/mk8574/birdclef_2023/data/test_soundscapes'

    # model checkpoint(s), not loading currently
    model_ckpt = None
    model_path = None
    
    # audio sampling data
    SR = 32000
    DURATION = 5
    MAX_READ_SAMPLES = 5
    LR = 5e-4


# In[4]:


pl.seed_everything(Config.seed, workers = True)


# In[5]:


# return config as dictionary
def config_to_dict(cfg):
    return dict((name, getattr(cfg, name)) for name in dir(cfg) if not name.startswith('__'))


# In[6]:


# read metadata
df = pd.read_csv('./train_metadata.csv')
df['secondary_labels'] = df['secondary_labels'].apply(lambda x: re.findall(r"'(\w+)'", x))
df['len_sec_labels'] = df['secondary_labels'].map(len)


# In[ ]:


# df.primary_label.value_counts()


# In[8]:


df_test = pd.DataFrame(
    [(path.stem, *path.stem.split("_"), path) for path in Path(Config.test_path).glob("*.ogg")],
    columns = ['filename', 'name', 'id', 'path']
)

df_test


# In[9]:


from sklearn.model_selection import train_test_split
import pandas as pd

def birds_stratified_split(df, target_col, test_size=0.2):
    class_counts = df[target_col].value_counts()
    low_count_classes = class_counts[class_counts < 2].index.tolist() ### Birds with single counts

    df['train'] = df[target_col].isin(low_count_classes)

    train_df, val_df = train_test_split(df[~df['train']], test_size=test_size, stratify=df[~df['train']][target_col], random_state=42)

    train_df = pd.concat([train_df, df[df['train']]], axis=0).reset_index(drop=True)

    # Remove the 'valid' column
    train_df.drop('train', axis=1, inplace=True)
    val_df.drop('train', axis=1, inplace=True)

    return train_df, val_df


# In[10]:


train_df, valid_df = birds_stratified_split(df, 'primary_label', 0.2)


# In[11]:


Path('specs/train').mkdir(exist_ok = True, parents = True)
Path('specs/valid').mkdir(exist_ok = True, parents = True)


# In[12]:


def get_audio(path):
    
    with SoundFile(path) as f:
        sr = f.samplerate
        
    return {'sr': sr}


# In[13]:


# adds "path" label to df
def add_path_df(df):
    
    df["path"] = [str(Path("./train_audio")/filename) for filename in df.filename]
    df = df.reset_index(drop=True)
    pool = joblib.Parallel(2)
    mapper = joblib.delayed(get_audio)
    tasks = [mapper(filepath) for filepath in df.path]
    df2 =  pd.DataFrame(pool(tqdm(tasks))).reset_index(drop=True)
    df = pd.concat([df,df2], axis=1).reset_index(drop=True)

    return df


# In[14]:


train_df = add_path_df(train_df)


# In[15]:


valid_df = add_path_df(valid_df)


# In[16]:


def compute_melspec(y, sr, n_mels, fmin, fmax):
    """
    Computes a mel-spectrogram and puts it at decibel scale
    Arguments:
        y {np array} -- signal
        params {AudioParams} -- Parameters to use for the spectrogram. Expected to have the attributes sr, n_mels, f_min, f_max
    Returns:
        np array -- Mel-spectrogram
    """
    melspec = lb.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, fmin=fmin, fmax=fmax,
    )

    melspec = lb.power_to_db(melspec).astype(np.float32)
    return melspec


# In[17]:


def mono_to_color(X, eps=1e-6, mean=None, std=None):
    mean = mean or X.mean()
    std = std or X.std()
    X = (X - mean) / (std + eps)
    
    _min, _max = X.min(), X.max()

    if (_max - _min) > eps:
        V = np.clip(X, _min, _max)
        V = 255 * (V - _min) / (_max - _min)
        V = V.astype(np.uint8)
    else:
        V = np.zeros_like(X, dtype=np.uint8)

    return V

def crop_or_pad(y, length, is_train=True, start=None):
    if len(y) < length:
        y = np.concatenate([y, np.zeros(length - len(y))])
        
        n_repeats = length // len(y)
        epsilon = length % len(y)
        
        y = np.concatenate([y]*n_repeats + [y[:epsilon]])
        
    elif len(y) > length:
        if not is_train:
            start = start or 0
        else:
            start = start or np.random.randint(len(y) - length)

        y = y[start:start + length]

    return y


# In[18]:


class AudioToImage:
    def __init__(self, sr=Config.SR, n_mels=128, fmin=0, fmax=None, duration=5, step=None, res_type="kaiser_fast", resample=True, train = True):

        self.sr = sr
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or self.sr//2

        self.duration = duration
        self.audio_length = self.duration*self.sr
        self.step = step or self.audio_length
        
        self.res_type = res_type
        self.resample = resample

        self.train = train

    def audio_to_image(self, audio):
        melspec = compute_melspec(audio, self.sr, self.n_mels, self.fmin, self.fmax) 
        image = mono_to_color(melspec)
#         compute_melspec(y, sr, n_mels, fmin, fmax)
        return image

    def __call__(self, row, save=True):
        audio, orig_sr = sf.read(row.path, dtype="float32")

        if self.resample and orig_sr != self.sr:
            audio = lb.resample(audio, orig_sr, self.sr, res_type=self.res_type)

        audios = [audio[i:i+self.audio_length] for i in range(0, max(1, len(audio) - self.audio_length + 1), self.step)]
        audios[-1] = crop_or_pad(audios[-1] , length=self.audio_length)
        images = [self.audio_to_image(audio) for audio in audios]
        images = np.stack(images)

        if save:
            if self.train:
                path = Path('specs/train')/f"{row.filename}.npy"
            else:
                path = Path('specs/valid')/f"{row.filename}.npy"

            path.parent.mkdir(exist_ok=True, parents=True)
            np.save(str(path), images)
        else:
            return row.filename, images


# In[19]:


def get_audios_as_images(df, train = True):
    pool = joblib.Parallel(2)
    
    converter = AudioToImage(step=int(5*0.666*Config.SR),train=train)
    mapper = joblib.delayed(converter)
    tasks = [mapper(row) for row in df.itertuples(False)]
    pool(tqdm(tasks))


# In[20]:


tqdm.pandas()


# In[21]:


# get_audios_as_images(train_df, train = True)


# In[22]:


# get_audios_as_images(valid_df, train = False)


# In[23]:


# df_train = pd.read_csv(Config.train_path)
# df_valid = pd.read_csv(Config.valid_path)
# df_train.head()


# In[24]:


birds = list(train_df.primary_label.unique())
missing_birds = list(set(list(train_df.primary_label.unique())).difference(list(valid_df.primary_label.unique())))
non_missing_birds = list(set(list(train_df.primary_label.unique())).difference(missing_birds))
len(non_missing_birds)
valid_df[missing_birds] = 0
valid_df = valid_df[train_df.columns] ## Fix order
# train_df


# In[25]:


# df['secondary_labels'] = df['secondary_labels'].apply(lambda x: re.findall(r"'(\w+)'", x))

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(birds)

train_df['labels'] = train_df['primary_label'].apply(lambda x: le.transform([x])[0])
valid_df['labels'] = valid_df['primary_label'].apply(lambda x: le.transform([x])[0])


# In[26]:


import albumentations as A

def get_train_transform():
    return A.Compose([
        A.HorizontalFlip(p = 0.5),
        A.OneOf([
            A.Cutout(max_h_size = 5, max_w_size = 16),
            A.CoarseDropout(max_holes = 4)
        ], p = 0.5)
    ])


# In[73]:


# defines the dataset, inheriting from torch.Dataset
class BirdDataset(torch.utils.data.Dataset):
    
    # saves from df, access path from df data
    def __init__(self, df, sr = Config.SR, duration = Config.DURATION, augmentations = None, mode = 'train'):        
        self.df = df
        self.sr = sr
        self.mode = mode
        self.duration = duration
        self.augmentations = augmentations
        
        
        if mode == 'train':
            self.img_dir = './specs/train'
            
        elif mode == 'valid':
            self.img_dir = './specs/valid'
            
        elif mode == 'test':
            self.img_dir = './test_soundscapes'
            
            
    def __len__(self):
        return len(self.df)


    @staticmethod
    def normalize(image):
        image = image / 255.0

        return image


    def __getitem__(self, idx):
        row = self.df.iloc[idx] # save row corresponding to df[:][idx]

        impath = os.path.join(self.img_dir, f"{row.filename}.npy") # extract image path from row

        image = np.load(str(impath))[:Config.MAX_READ_SAMPLES] # load spectogram from impath

        if self.mode == 'train':
            image = image[np.random.choice(len(image))]

        else:
            image = image[0]

        if self.augmentations:
            image = self.augmentations(image = image)
            image = image['image']

        image = torch.tensor(image).float()
            
        image.size()

        image = torch.stack([image, image, image])

        image = self.normalize(image)

        return image, row['labels']


# In[74]:


def get_fold_dls(df_train, df_valid):
    
    ds_train = BirdDataset(
        df_train,
        sr = Config.SR,
        duration = Config.DURATION,
        augmentations = A.Compose([
            A.GaussNoise(p = 1),
            A.Blur(p = 0.5),
            A.VerticalFlip(p = 0.5)
        ]),
        mode = 'train'
    )
    
    ds_val = BirdDataset(
        df_valid,
        sr = Config.SR,
        duration = Config.DURATION,
        augmentations = None,
        mode = 'valid'
    )
    
    dl_train = DataLoader(ds_train, batch_size = Config.batch_size, shuffle = True, num_workers = 2)
    dl_val = DataLoader(ds_val, batch_size = Config.batch_size, shuffle = True, num_workers = 2)
    
    return dl_train, dl_val, ds_train, ds_val


# In[75]:


_, _, ds_train, _ = get_fold_dls(train_df, valid_df)

x, y = ds_train[0]


# In[76]:


def show_batch(img_ds, num_items, num_rows, num_cols, predict_arr=None):
    fig = plt.figure(figsize=(12, 6))    
    img_index = np.random.randint(0, len(img_ds)-1, num_items)
    for index, img_index in enumerate(img_index):  # list first 9 images
        img, lb = img_ds[img_index]        
        ax = fig.add_subplot(num_rows, num_cols, index + 1, xticks=[], yticks=[])
        if isinstance(img, torch.Tensor):
            img = img.detach().numpy()
        if isinstance(img, np.ndarray):
            img = img.transpose(1, 2, 0)
            ax.imshow(img)        
            
        title = f"Spec"
        ax.set_title(title)


# In[78]:


dl_train, dl_val, ds_train, ds_val = get_fold_dls(train_df, valid_df)
show_batch(ds_train, 8, 2, 4)


# In[79]:


from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau, OneCycleLR

def get_optimizer(lr, params):
    model_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, params),
        lr = lr,
        weight_decay = Config.weight_decay
    )
    
    interval = "epoch"
    lr_scheduler = CosineAnnealingWarmRestarts(
        model_optimizer,
        T_0 = Config.epochs,
        T_mult = 1,
        eta_min = 1e-6, 
        last_epoch = -1
    )
    
    return {
        'optimizer': model_optimizer,
        'lr_scheduler': {
            'scheduler': lr_scheduler,
            'interval': interval,
            'monitor': 'val_loss',
            'frequency': 1
        }
    }


# In[80]:


from torchtoolbox.tools import mixup_data, mixup_criterion
import torch.nn as nn
from torch.nn.functional import cross_entropy
import torchmetrics
import timm


# In[81]:


import sklearn.metrics

def padded_cmap(solution, submission, padding_factor=5):
    solution = solution#.drop(['row_id'], axis=1, errors='ignore')
    submission = submission#.drop(['row_id'], axis=1, errors='ignore')
    new_rows = []
    for i in range(padding_factor):
        new_rows.append([1 for i in range(len(solution.columns))])
    new_rows = pd.DataFrame(new_rows)
    new_rows.columns = solution.columns
    padded_solution = pd.concat([solution, new_rows]).reset_index(drop=True).copy()
    padded_submission = pd.concat([submission, new_rows]).reset_index(drop=True).copy()
    score = sklearn.metrics.average_precision_score(
        padded_solution.values,
        padded_submission.values,
        average='macro',
    )
    return score

def map_score(solution, submission):
    solution = solution #.drop(['row_id'], axis=1, errors='ignore')
    submission = submission #.drop(['row_id'], axis=1, errors='ignore')
    score = sklearn.metrics.average_precision_score(
        solution.values,
        submission.values,
        average='micro',
    )
    return score


# In[82]:


# valid_df


# In[83]:


# dummy = valid_df[birds].copy()
# dummy[birds] = np.random.rand(dummy.shape[0],dummy.shape[1])


# In[84]:


class BirdClefModel(pl.LightningModule):
    def __init__(self, model_name=Config.model, num_classes = Config.num_classes, pretrained = Config.pretrained):
        super().__init__()
        self.num_classes = num_classes

        self.backbone = timm.create_model(model_name, pretrained=pretrained)

        if 'res' in model_name:
            self.in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(self.in_features, num_classes)
        elif 'dense' in model_name:
            self.in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Linear(self.in_features, num_classes)
        elif 'efficientnet' in model_name:
            self.in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Sequential(
                nn.Linear(self.in_features, num_classes)
            )
        
        self.loss_function = nn.CrossEntropyLoss() # nn.BCELoss fuck
        self.validation_step_outputs = {'val_loss': [], 'logits': [], 'targets': []}

        self.save_hyperparameters()
        
    def forward(self,images):
        logits = self.backbone(images)
        return logits
    
    def configure_optimizers(self):
        return get_optimizer(lr=Config.LR, params=self.parameters())

    def train_with_mixup(self, X, y):
        X, y_a, y_b, lam = mixup_data(X, y, alpha=Config.mixup_alpha)
        y_pred = self(X)
        loss_mixup = mixup_criterion(cross_entropy, y_pred, y_a, y_b, lam)
        return loss_mixup

    def training_step(self, batch, batch_idx):
        image, target = batch        
        if Config.use_mixup:
            loss = self.train_with_mixup(image, target)
        else:
            y_pred = self(image)
            loss = self.loss_function(y_pred,target)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss        

    def validation_step(self, batch, batch_idx):
        image, target = batch     
        y_pred = self(image)
        val_loss = self.loss_function(y_pred, target)
        self.log("val_loss", val_loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)

        self.validation_step_outputs['val_loss'].append(val_loss)
        self.validation_step_outputs['logits'].append(y_pred)
        self.validation_step_outputs['targets'].append(target)
        
        return {"val_loss": val_loss, "logits": y_pred, "targets": target}
    
    def train_dataloader(self):
        return self._train_dataloader 
    
    def validation_dataloader(self):
        return self._validation_dataloader
    
    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        avg_loss = torch.stack(outputs['val_loss']).mean()
        output_val = torch.cat(outputs['logits'], dim=0).sigmoid().cpu().detach().numpy()
        target_val = torch.cat(outputs['targets'], dim=0).cpu().detach().numpy()

#         val_df = pd.DataFrame(target_val, columns = birds)
#         pred_df = pd.DataFrame(output_val, columns = birds)
        
#         avg_score = padded_cmap(val_df, pred_df, padding_factor = 5)
#         avg_score2 = padded_cmap(val_df, pred_df, padding_factor = 3)
#         avg_score3 = sklearn.metrics.label_ranking_average_precision_score(target_val,output_val)
        avg_score = avg_loss
    
        print(f'epoch {self.current_epoch} validation loss {avg_loss}')
#         print(f'epoch {self.current_epoch} validation C-MAP score pad 5 {avg_score}')
#         print(f'epoch {self.current_epoch} validation C-MAP score pad 3 {avg_score2}')
#         print(f'epoch {self.current_epoch} validation AP score {avg_score3}')
        
        
        # val_df.to_pickle('val_df.pkl')
        # pred_df.to_pickle('pred_df.pkl')
        
        
        return {'val_loss': avg_loss, 'val_cmap':avg_score}


# In[85]:


import gc

def run_training():
    print(f"Running training...")
    logger = None
    
    
    dl_train, dl_val, ds_train, ds_val = get_fold_dls(train_df, valid_df)
    
    audio_model = BirdClefModel()

#     early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=Config.PATIENCE, verbose= True, mode="min")
#     checkpoint_callback = ModelCheckpoint(monitor='val_loss',
#                                           dirpath= "/kaggle/working/exp1/",
#                                       save_top_k=1,
#                                       save_last= True,
#                                       save_weights_only=True,
#                                       filename= f'./{Config.model}_loss',
#                                       verbose= True,
#                                       mode='min')
    
#     callbacks_to_use = [checkpoint_callback, early_stop_callback]


    trainer = pl.Trainer(
        val_check_interval=0.5,
        deterministic=True,
        max_epochs=Config.epochs,
        logger=logger,
        precision=Config.PRECISION, 
        default_root_dir = './model/',
        accelerator="gpu"
    )

    print("Running trainer.fit")
    trainer.fit(audio_model, train_dataloaders = dl_train, val_dataloaders = dl_val)                

    gc.collect()
    torch.cuda.empty_cache()


# In[ ]:


run_training()


# In[40]:


def run_inference(path = None):
    def predict(data_loader, model):
        model.to('cpu')
        model.eval()

        predictions = []
        for ex in range(len(ds_test)):
            images = torch.from_numpy(ds_test[ex])

            with torch.no_grad():
                outputs = model(images).sigmoid().detach().cpu().numpy()

            predictons.append(outputs)
        return predictions

    ds_test = BirdDataset(
        df_test, 
        sr = Config.SR,
        duration = Config.DURATION,
    )
    
    if path is None:
        path = Config.model_ckpt

    model = BirdClefModel.load_from_checkpoint(path, train_dataloader = None, validation_dataloader = None)
    preds = predict(ds_test, model)

    gc.collect()
    torch.cuda.empty_cache()


# In[ ]:


df_test


# In[ ]:


run_inference('/scratch/network/mk8574/birdclef_2023/data/model/lightning_logs/version_1949292/checkpoints/epoch=9-step=2120.ckpt')


# In[ ]:




