import torch
import numpy as np
from tqdm.auto import tqdm
from time import time
from torch.cuda.amp import autocast, GradScaler

def validation(model, valid_loader, creterion, device):
    model.eval()
    val_loss = []

    total, correct = 0, 0

    with torch.no_grad():
        for waveforms, labels in tqdm(iter(valid_loader)):
            waveforms = waveforms.to(device)
            labels = labels.to(device)

            output = model(waveforms)            
            loss = creterion(output, labels)

            val_loss.append(loss.item())

            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += predicted.eq(labels).cpu().sum()

    accuracy = correct / total

    avg_loss = np.mean(val_loss)

    return avg_loss, accuracy

def train(model, creterion, train_loader, valid_loader, optimizer, scheduler, recorder, config, amp=False):
    device = config.device
    
    accumulation_step = int(config.total_batch_size / config.batch_size)
    if amp:
        scaler = GradScaler()
    model.to(device)

    best_model = None
    best_acc = 0
    
    for epoch in range(1, config.max_epoch+1):
        train_loss = []
        model.train()
        
        row = dict()
        row['epoch'] = epoch
        row['lr'] = optimizer.param_groups[0]['lr']
        
        tic = time()
        for i, (waveforms, labels) in enumerate(tqdm(train_loader)):
            waveforms = waveforms.to(device)
            labels = labels.flatten().to(device)
            
            if amp:
                with autocast():
                    optimizer.zero_grad()
                    output = model(waveforms)
                    loss = creterion(output, labels)
                scaler.scale(loss).backward()
            
                if (i+1) % accumulation_step == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else: 
                optimizer.zero_grad()
            
                output = model(waveforms)
                loss = creterion(output, labels)
                loss.backward()

                if (i+1) % accumulation_step == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            train_loss.append(loss.item())
        toc = time()
        
        row['train_elapsed_time'] = round(toc-tic, 1)
        
        avg_loss = np.mean(train_loss)
        tic = time()
        valid_loss, valid_acc = validation(model, valid_loader, creterion, device)
        toc = time()
        
        row['valid_elapsed_time'] = round(toc-tic, 1)        
        row['train_loss'] = avg_loss
        row['valid_loss'] = valid_loss
        row['valid_accuracy'] = valid_acc.item()
        
        recorder.add_row(row)
        recorder.save_plot()
        
        if scheduler is not None:
            scheduler.step(valid_loss)

        if valid_acc > best_acc:
            best_acc = valid_acc
            best_model = model

        print(f'epoch:[{epoch}] train loss:[{avg_loss:.5f}] valid_loss:[{valid_loss:.5f}] valid_acc:[{valid_acc:.5f}]')
    
    print(f'best_acc:{best_acc:.5f}')

    return best_model