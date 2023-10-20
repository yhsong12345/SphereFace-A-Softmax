import torch
import matplotlib.pyplot as plt
import os
import pandas as pd



            

plt.style.use('ggplot')

def save_plots(train_acc, valid_acc, train_loss, valid_loss):
    path = f'./outputs'
    plt.figure(figsize=(10,7))
    plt.plot(
        train_acc, color='red', linestyle='-',
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-',
        label='validation accuracy'
    )

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'{path}/accuracy.png')


    plt.figure(figsize=(10,7))
    plt.plot(
        train_loss, color='green', linestyle='--',
        label='train loss'
    )
    plt.plot(
        valid_loss, color='orange', linestyle='--',
        label='validation loss'
    )

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{path}/loss.png')




def plot_features(features, labels, num_classes, epoch, prefix):
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    for i in range(num_classes):
        plt.scatter(
            features[labels==i, 0],
            features[labels==i, 1],
            c = colors[i],
            s=.1
        )
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
    if prefix:
        dir = './plots/train'
        if not os.path.exists(dir):
            os.mkdir(dir)
        plt.savefig(f'{dir}/{epoch+1}.png')
    else:
        dir ='./plots/test'
        if not os.path.exists(dir):
            os.mkdir(dir)
        plt.savefig(f'{dir}/{epoch+1}.png')