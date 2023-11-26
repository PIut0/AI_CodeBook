from matplotlib import pyplot as plt
import pandas as pd
import logging
import torch
import csv
import os


class Recorder():
    def __init__(self, record_dir: str):
        self.record_dir = record_dir
        self.plot_dir = os.path.join(record_dir, 'plots')
        self.record_filepath = os.path.join(self.record_dir, 'record.csv')

        os.makedirs(self.plot_dir, exist_ok=True)

    def add_row(self, row_dict: dict):
        fieldnames = list(row_dict.keys())

        with open(self.record_filepath, newline='', mode='a') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if f.tell() == 0:
                writer.writeheader()

            writer.writerow(row_dict)
    
    def save_plot(self):
        record_df = pd.read_csv(self.record_filepath)
        current_epoch = record_df['epoch'].max()
        epoch_range = list(range(1, current_epoch+1))
        
        fig = plt.figure(figsize=(20, 8))
        values = record_df['lr'].tolist()
        plt.plot(epoch_range, values, marker='.', c='red', label='lr')
        self.save_plot_func(fig, 'learning rate', epoch_range)
        
        fig = plt.figure(figsize=(20, 8))
        values = record_df['valid_accuracy'].tolist()
        plt.plot(epoch_range, values, marker='.', c='blue', label='accuracy')
        self.save_plot_func(fig, 'accuracy', epoch_range)
        
        fig = plt.figure(figsize=(20, 8))
        values = record_df['train_loss'].tolist()
        plt.plot(epoch_range, values, marker='.', c='red', label='train_loss')
        values = record_df['valid_loss'].tolist()
        plt.plot(epoch_range, values, marker='.', c='blue', label='valid_loss')
        self.save_plot_func(fig, 'loss', epoch_range)
        
        
    def save_plot_func(self, fig, plot_name, epoch_range):
        plt.title(plot_name, fontsize=15)
        plt.legend(loc='upper right')
        plt.grid()
        plt.xlabel('epoch')
        plt.ylabel(plot_name)
        plt.xticks(epoch_range, [str(i) for i in epoch_range])
        plt.close(fig)
        fig.savefig(os.path.join(self.plot_dir, plot_name+'.png'))