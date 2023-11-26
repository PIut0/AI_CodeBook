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

transformers.logging.set_verbosity_error()


@dataclass
class Config:
    # data args
    train_csv: str = "/scratch/network/mk8574/audio_sentiment_challenge/data/train.csv"
    test_csv: str = "/scratch/network/mk8574/audio_sentiment_challenge/data/test.csv"

    # model args
    # pretrained_name: str = "Rajaram1996/Hubert_emotion"
    pretrained_name: str = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
    
    train_serial = datetime.now().strftime("%Y%m%d_%H%M%S") + "|" + pretrained_name.replace("/", "|")

    # k-fold
    k_fold_num: int = 5  # if you want to use k-fold validation, set positive integer value.
    k_fold_idx: int = 1

    # save dir
    save_dir: str = f"/scratch/network/mk8574/audio_sentiment_challenge/baseline_lks/results/{train_serial}/"

    # hparams
    seed: int = 42
    lr: float = 5e-4
    batch_size: int = 10
    gradient_accumulate_step: int = 4  # total batch size = batch_size * gradient_accumulate_step
    max_epoch: int = batch_size * gradient_accumulate_step
    early_stop_patience = 5
    
config = Config()

if not os.path.exists(config.save_dir):
    os.makedirs(config.save_dir)

with open(os.path.join(config.save_dir, "config.json"), "w") as config_file:
    json.dump(asdict(config), config_file, indent=4, sort_keys=False)
    
seed = config.seed

random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

def load_pretrained(pretrained_name: str) -> Tuple[AutoModelForAudioClassification, AutoFeatureExtractor]:
    # model
    model = AutoModelForAudioClassification.from_pretrained(pretrained_name)

    model.config.num_labels = 6
    model.classifier = nn.Linear(in_features=model.projector.out_features, out_features=6)
    nn.init.kaiming_normal_(model.classifier.weight, mode="fan_in", nonlinearity="relu")
    nn.init.zeros_(model.classifier.bias)

    # feature extractor
    feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_name)

    return model, feature_extractor

class DataModule:
    def __init__(
        self,
        feature_extractor: AutoFeatureExtractor,
        transforms: list = None,
    ):
        self.feature_extractor = feature_extractor
        self.transforms = transforms

    def to_dataset(self, df: pd.DataFrame) -> Dataset:
        def load_waveform(row):
            waveform, sample_rate = librosa.load(row["path"])
            waveform = librosa.resample(
                waveform,
                orig_sr=sample_rate,
                target_sr=self.feature_extractor.sampling_rate,
            )
            row["waveform"] = waveform

            return row

        dataset = Dataset.from_pandas(df)
        dataset = dataset.map(load_waveform, num_proc=4)
        if "label" in dataset.column_names:
            dataset = dataset.rename_column("label", "labels")

        return dataset

    def apply_transforms(self, dataset: Dataset) -> Dataset:
        def apply_transforms(batch):
            waveforms = [self.transforms(samples=np.array(waveform, dtype=np.float32), sample_rate=self.feature_extractor.sampling_rate) for waveform in batch["waveform"]]
            batch["waveform"] = waveforms

            return batch

        if self.transforms:
            dataset = dataset.with_transform(apply_transforms)

        return dataset

    def collate_fn(self, batch: list) -> BatchFeature:
        if hasattr(self.feature_extractor, "nb_max_frames"):
            padding = "max_length"
        else:
            padding = "longest"

        waveforms = [data["waveform"] for data in batch]
        model_inputs = self.feature_extractor(
            waveforms,
            sampling_rate=self.feature_extractor.sampling_rate,
            padding=padding,
            return_tensors="pt",
        )

        if "labels" in batch[0]:
            labels = [data["labels"] for data in batch]
            model_inputs["labels"] = torch.tensor(labels)

        return model_inputs
    
class MetricScore(nn.Module):
    def __init__(self):
        super().__init__()

        metrics = MetricCollection(
            {
                "accuracy": MulticlassAccuracy(num_classes=6),
                "recall": MulticlassRecall(num_classes=6),
                "precision": MulticlassPrecision(num_classes=6),
                "f1": MulticlassF1Score(num_classes=6),
            }
        )
        self.train_metrics = metrics.clone()
        self.valid_metrics = metrics.clone()

        self.train_losses = []
        self.valid_losses = []

    def add_train_metrics(self, logits: torch.Tensor, labels: torch.Tensor):
        self.train_metrics.update(logits, labels)

    def add_valid_metrics(self, logits: torch.Tensor, labels: torch.Tensor):
        self.valid_metrics.update(logits, labels)

    def add_train_loss(self, loss: torch.Tensor):
        self.train_losses.append(loss.item())

    def add_valid_loss(self, loss: torch.Tensor):
        self.valid_losses.append(loss.item())

    def compute_train(self) -> Dict[str, Any]:
        scores = self.train_metrics.compute()
        for metric_key, score in scores.items():
            if isinstance(score, torch.Tensor):
                scores[metric_key] = score.item()
        scores.update({"loss": np.mean(self.train_losses)})

        return scores

    def compute_valid(self) -> Dict[str, Any]:
        scores = self.valid_metrics.compute()
        for metric_key, score in scores.items():
            if isinstance(score, torch.Tensor):
                scores[metric_key] = score.item()
        scores.update({"loss": np.mean(self.valid_losses)})

        return scores

    def reset(self):
        self.train_metrics.reset()
        self.valid_metrics.reset()

        self.train_losses = []
        self.valid_losses = []

    def print_summary(self, epoch_idx: int):
        train_result = self.compute_train()
        valid_result = self.compute_valid()

        assert list(train_result.keys()) == list(valid_result.keys())

        pt = PrettyTable()
        pt.field_names = [f"epoch {epoch_idx}"] + list(train_result.keys())

        train_row = ["train"]
        for score in train_result.values():
            train_row.append(round(score, 3))
        pt.add_row(train_row)

        valid_row = ["valid"]
        for score in valid_result.values():
            valid_row.append(round(score, 3))
        pt.add_row(valid_row)

        print(pt, end="\n\n")
        
        
def fit(
    model: AutoModelForAudioClassification,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    max_epoch: int = 64,
    lr: float = 5e-4,
    gradient_accumulate_step: int = 1,
    early_stop_patience: int = 5,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    metric_scores = MetricScore().to(device)

    optimizer = AdamW(
        [{"params": module.parameters(), "lr": lr if name == "classifier" else lr * 0.1} for name, module in model.named_children()],
        weight_decay=0.1,
    )
    lr_scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    # run finetune
    best_score = 0.0
    best_state_dict = None

    early_stop_count = 0
    for epoch_idx in range(1, max_epoch):
        with torch.set_grad_enabled(True), tqdm(total=len(train_loader), desc=f"[Epoch {epoch_idx}/{max_epoch}] training", leave=False) as pbar:
            model.train()
            for step_idx, batch in enumerate(train_loader, 1):
                batch = batch.to(device)

                output = model(**batch)
                loss = output.loss
                loss.backward()

                if step_idx % gradient_accumulate_step == 0 or step_idx == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad()            

                metric_scores.add_train_loss(loss)
                metric_scores.add_train_metrics(output.logits, batch.labels)

                pbar.update()
                pbar.set_postfix({"train loss": loss.item()})

        with torch.set_grad_enabled(False), tqdm(total=len(valid_loader), desc=f"[Epoch {epoch_idx}/{max_epoch}] validation", leave=False) as pbar:
            model.eval()
            for batch in valid_loader:
                batch = batch.to(device)

                output = model(**batch)
                loss = output.loss

                metric_scores.add_valid_loss(loss)
                metric_scores.add_valid_metrics(output.logits, batch.labels)

                pbar.update()
                pbar.set_postfix({"valid loss": loss.item()})

        epoch_score = (metric_scores.compute_valid()["accuracy"] * 0.8 + metric_scores.compute_train()["accuracy"] * 0.2)
        metric_scores.print_summary(epoch_idx=epoch_idx)
        metric_scores.reset()

        lr_scheduler.step(epoch_score)

        if epoch_score > best_score:
            best_score = epoch_score
            best_state_dict = model.state_dict()
            early_stop_count = 0

            model.load_state_dict(best_state_dict)
            model.save_pretrained(os.path.join(config.save_dir, f"epoch{epoch_idx}_{epoch_score}"))
            feature_extractor.save_pretrained(os.path.join(config.save_dir, f"epoch{epoch_idx}_{epoch_score}"))
        else:
            early_stop_count += 1
            if early_stop_count == early_stop_patience:
                print("*** EARLY STOPPED ***")
                break
    
    return best_state_dict

@torch.no_grad()
def predict(
    model: AutoModelForAudioClassification,
    feature_extractor: AutoFeatureExtractor,
    test_dataset: Dataset,
    batch_size: int = 16,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    predict_logits = {"id": []}
    predict_logits.update({class_id: [] for class_id in range(6)})

    predict_class = {"id": [], "label": []}

    for batch_idx in tqdm(range(0, len(test_dataset), batch_size), desc="prediction"):
        bs, bi = batch_idx, batch_idx + batch_size
        batch = test_dataset[bs:bi]

        if hasattr(feature_extractor, "nb_max_frames"):
            padding = "max_length"
        else:
            padding = "longest"

        model_inputs = feature_extractor(
            batch["waveform"],
            sampling_rate=feature_extractor.sampling_rate,
            padding=padding,
            return_tensors="pt",
        ).to(device)

        model_output = model(**model_inputs)

        batch_logits = model_output.logits.cpu()
        batch_predict = model_output.logits.argmax(dim=-1).cpu()

        predict_logits["id"] += batch["id"]
        for class_id in range(6):
            predict_logits[class_id] += batch_logits[:, class_id].tolist()

        predict_class["id"] += batch["id"]
        predict_class["label"] += batch_predict.tolist()

    predict_logits = pd.DataFrame(predict_logits)
    predict_class = pd.DataFrame(predict_class)

    return predict_logits, predict_class

transforms = audiomentations.OneOf(
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
    ]
)

model, feature_extractor = load_pretrained(config.pretrained_name)

# create train data and valid data
df = pd.read_csv(config.train_csv)
df["path"] = df["path"].apply(lambda x: os.path.join(os.path.dirname(config.train_csv), x[2:]))

if config.k_fold_num > 0:
    skf = StratifiedKFold(n_splits=config.k_fold_num)
    train_indices, valid_indices = list(skf.split(df, df["label"]))[config.k_fold_idx]
    train_df, valid_df = df.iloc[train_indices], df.iloc[valid_indices]
else:
    train_df, valid_df = train_test_split(df, train_size=0.9, stratify=df["label"], random_state=seed)

# create test data
test_df = pd.read_csv(config.test_csv)
test_df["path"] = test_df["path"].apply(lambda x: os.path.join(os.path.dirname(config.test_csv), x[2:]))

data_module = DataModule(
    feature_extractor=feature_extractor,
    transforms=transforms,
)
train_dataset = data_module.apply_transforms(data_module.to_dataset(train_df))
valid_dataset = data_module.to_dataset(valid_df)

train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    collate_fn=data_module.collate_fn,
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    collate_fn=data_module.collate_fn,
)

best_state_dict = fit(
    model=model,
    train_loader=train_loader,
    valid_loader=valid_loader,
    max_epoch=config.max_epoch,
    lr=config.lr,
    gradient_accumulate_step=config.gradient_accumulate_step,
    early_stop_patience=config.early_stop_patience,
)

model.load_state_dict(best_state_dict)
model.save_pretrained(os.path.join(config.save_dir, "best_model"))
feature_extractor.save_pretrained(os.path.join(config.save_dir, "best_model"))

test_dataset = data_module.to_dataset(test_df)
predict_logits, predict_class = predict(model, feature_extractor, test_dataset, config.batch_size)

predict_logits.to_csv(os.path.join(config.save_dir, "predict_logits.csv"), index=False)
predict_class.to_csv(os.path.join(config.save_dir, "predict_class.csv"), index=False)