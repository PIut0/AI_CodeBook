from config import Config
from modules.trainer import train
from modules.model import MyModel
from modules.dataset import MyDataSet
from modules.dataloader import collate_fn_no_label

import sys, os
import torch
import pandas as pd
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor

def main(arg):
    if len(arg)==0:
        print('type train serial !!')
        sys.exit()
    
    config = Config()
    device = torch.device(f'cuda:{config.device}') if torch.cuda.is_available() else torch.device('cpu')
    
    working_path = os.path.join('/scratch/network/mk8574/audio_sentiment_challenge/baseline_dy/baseline/results', arg)
    model = MyModel(working_path, mode='test').to(device)
    
    test_df = pd.read_csv(os.path.join(config.data_path, 'test.csv'))
    test_df.reset_index(drop=True, inplace=True)
    
    feature_extractor = AutoFeatureExtractor.from_pretrained(config.pretrained_name)
    test_dataset = MyDataSet(test_df, feature_extractor, mode='test', data_path=config.data_path)

    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn_no_label, num_workers=config.num_workers)
    
    model.eval()
    preds = []

    with torch.no_grad():
        for x in tqdm(iter(test_loader)):
            x = x.to(device)
            output = model(x)
            
            preds += output.argmax(-1).detach().cpu().numpy().tolist()
    
    submission = pd.read_csv(os.path.join(config.data_path, 'sample_submission.csv'))
    submission['label'] = preds
    submission.to_csv(os.path.join(working_path, 'submission_2.csv'), index=False)
    
if __name__ == '__main__':
    main(sys.argv[1])
    print('End inference, submission file was created')