"""Train
"""
from datetime import datetime
from time import time
import numpy as np
import shutil, random, os, sys, torch
from glob import glob
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

prj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(prj_dir)

from modules.utils import load_yaml, get_logger
from modules.metrics import get_metric_function
from modules.earlystoppers import EarlyStopper
from modules.losses import get_loss_function
from modules.optimizers import get_optimizer
from modules.schedulers import get_scheduler
from modules.scalers import get_image_scaler
from modules.datasets import SegDataset
from modules.recorders import Recorder
from modules.trainer import Trainer
from models.utils import get_model

if __name__ == '__main__':
    
    # Load config
    config_path = os.path.join(prj_dir, 'config', 'train.yaml')
    config = load_yaml(config_path)
    
    # Set train serial: ex) 20211004
    if config['train_serial'] is not None:
        train_serial = config['train_serial']
        checkpoint_path = os.path.join(prj_dir, 'results', 'train', train_serial, 'model.pt')
    else:
        train_serial = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = None
    train_serial = 'debug' if config['debug'] else train_serial

    # Set random seed, deterministic
    torch.cuda.manual_seed(config['seed'])
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set device(GPU/CPU)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu_num'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create train result directory and set logger
    train_result_dir = os.path.join(prj_dir, 'results', 'train', train_serial)
    os.makedirs(train_result_dir, exist_ok=True)

    # Set logger
    logging_level = 'debug' if config['verbose'] else 'info'
    logger = get_logger(name='train',
                        file_path=os.path.join(train_result_dir, 'train.log'),
                        level=logging_level)


    # Set data directory
    train_dirs = os.path.join('/scratch/network/mk8574/crack_detection_challenge/data', 'train')

    # Load data and create dataset for train 
    # Load image scaler
    train_img_paths = glob(os.path.join(train_dirs, 'images', '*.jpg'))
    train_img_paths, val_img_paths = train_test_split(train_img_paths, test_size=config['val_size'], random_state=config['seed'], shuffle=True)

    train_dataset = SegDataset(paths=train_img_paths,
                            input_size=config['input_size'],
                            scaler=get_image_scaler(config['scaler']),
                            logger=logger)
    val_dataset = SegDataset(paths=val_img_paths,
                            input_size=config['input_size'],
                            scaler=get_image_scaler(config['scaler']),
                            logger=logger)
    # Create data loader
    train_dataloader = DataLoader(dataset=train_dataset,
                                batch_size=config['batch_size'],
                                num_workers=config['num_workers'], 
                                shuffle=config['shuffle'],
                                drop_last=config['drop_last'])
                                
    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=config['batch_size'],
                                num_workers=config['num_workers'], 
                                shuffle=False,
                                drop_last=config['drop_last'])

    logger.info(f"Load dataset, train: {len(train_dataset)}, val: {len(val_dataset)}")
    
    # Load model
    model = get_model(model_str=config['architecture'])
    model = model(classes=config['n_classes'],
                encoder_name=config['encoder'],
                encoder_weights=config['encoder_weight'],
                activation=config['activation']).to(device)
    logger.info(f"Load model architecture: {config['architecture']}")

    # Set optimizer
    optimizer = get_optimizer(optimizer_str=config['optimizer']['name'])
    optimizer = optimizer(model.parameters(), **config['optimizer']['args'])
    
    # Set Scheduler
    scheduler = get_scheduler(scheduler_str=config['scheduler']['name'])
    scheduler = scheduler(optimizer=optimizer, **config['scheduler']['args'])

    # Set loss function
    loss_func = get_loss_function(loss_function_str=config['loss']['name'])
    loss_func = loss_func(**config['loss']['args'])

    # Set metric
    metric_funcs = {metric_name:get_metric_function(metric_name) for metric_name in config['metrics']}
    logger.info(f"Load optimizer:{config['optimizer']['name']}, scheduler: {config['scheduler']['name']}, loss: {config['loss']['name']}, metric: {config['metrics']}")

    # Set trainer
    trainer = Trainer(model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    loss_func=loss_func,
                    metric_funcs=metric_funcs,
                    device=device,
                    logger=logger,
                    checkpoint_path=checkpoint_path
                    )
    logger.info(f"Load trainer")

    # Set early stopper
    early_stopper = EarlyStopper(patience=config['earlystopping_patience'],
                                logger=logger,
                                patience_counter=trainer.patience_counter,
                                best_target=trainer.best_target)
    # Set recorder
    recorder = Recorder(record_dir=train_result_dir,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        logger=logger,
                        )
    logger.info("Load early stopper, recorder")

    # Recorder - save train config
    shutil.copy(config_path, os.path.join(recorder.record_dir, 'train.yaml'))

    # Train
    print("START TRAINING")
    logger.info("START TRAINING")
    for epoch_id in range(trainer.start_epoch, config['n_epochs']):
        
        # Initiate result row
        row = dict()
        row['epoch_id'] = epoch_id
        row['train_serial'] = train_serial
        #row['lr'] = trainer.scheduler.get_last_lr()[0]

        # Train
        print(f"Epoch {epoch_id}/{config['n_epochs']} Train..")
        logger.info(f"Epoch {epoch_id}/{config['n_epochs']} Train..")
        tic = time()
        trainer.train(dataloader=train_dataloader, epoch_index=epoch_id)
        toc = time()
        # Write tarin result to result row
        row['train_loss'] = trainer.loss  # Loss
        for metric_name, metric_score in trainer.scores.items():
            row[f'train_{metric_name}'] = metric_score

        row['train_elapsed_time'] = round(toc-tic, 1)
        # Clear
        trainer.clear_history()

        # Validation
        print(f"Epoch {epoch_id}/{config['n_epochs']} Validation..")
        logger.info(f"Epoch {epoch_id}/{config['n_epochs']} Validation..")
        tic = time()
        trainer.validate(dataloader=val_dataloader, epoch_index=epoch_id)
        toc = time()
        row['val_loss'] = trainer.loss
        # row[f"val_{config['metric']}"] = trainer.score
        for metric_name, metric_score in trainer.scores.items():
            row[f'val_{metric_name}'] = metric_score
        row['val_elapsed_time'] = round(toc-tic, 1)
        trainer.clear_history()

        # Performance record - row
        recorder.add_row(row)
        
        # Performance record - plot
        recorder.save_plot(config['plot'])

        # Check early stopping
        early_stopper.check_early_stopping(row[config['earlystopping_target']])
        recorder.save_by_condition(epoch=epoch_id, patience_counter=early_stopper.patience_counter, best_target=early_stopper.best_target)
        if early_stopper.patience_counter == 0:
            recorder.save_weight(epoch=epoch_id, patience_counter=early_stopper.patience_counter, best_target=early_stopper.best_target)
            
        if early_stopper.stop:
            print(f"Epoch {epoch_id}/{config['n_epochs']}, Stopped counter {early_stopper.patience_counter}/{config['earlystopping_patience']}")
            logger.info(f"Epoch {epoch_id}/{config['n_epochs']}, Stopped counter {early_stopper.patience_counter}/{config['earlystopping_patience']}")
            break

    print("END TRAINING")
    logger.info("END TRAINING")