# -*-coding:utf-8-*-
# -*-coding:utf-8-*-
import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from dataloader import create_DTA_dataset
from model import Graph_GAT, Graph_GCN, Graph_noSub, Graph_noConn, Graph_noATT
from utils import *
import argparse
from log.train_logger import TrainLogger

# training function at each epoch
def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for data in train_loader:
        optimizer.zero_grad()
        data_mol = data[0].to(device)
        data_pro = data[1].to(device)
        data_clique = data[2].to(device)
        output = model(data_mol, data_pro, data_clique)
        loss = loss_fn(output, data_mol.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        # running_loss.update(loss.item(), data[1].y.size(0))
    print('Train epoch: {}\tLoss: {:.6f}'.format(epoch, loss.item()))

def predicting(model, device, dataloader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(dataloader.dataset)))
    with torch.no_grad():
        for data in dataloader:
            data_mol = data[0].to(device)
            data_pro = data[1].to(device)
            data_clique = data[2].to(device)
            output = model(data_mol, data_pro, data_clique)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels,data_mol.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()

datasets = [['davis', 'kiba'][0]]
cuda_name = ['cuda:0', 'cuda:1'][0]
# datasets = [['davis', 'kiba'][int(sys.argv[1])]]
# cuda_name = ['cuda:0', 'cuda:1'][int(sys.argv[2])]
modeling = [Graph_GAT][0]
model_st = modeling.__name__
BATCH_SIZE = 2
LR = 0.0005

params = dict(
    data_root="data",
    save_dir="save",
    dataset=datasets[0],
    save_model="save_model",
    lr=LR,
    batch_size=BATCH_SIZE,
    model_name=model_st
)
logger = TrainLogger(params)
logger.info(__file__)

# Main program: iterate over different datasets
for dataset in datasets:
    print('\nrunning on ', model_st + '_' + dataset )
    train_data, test_data= create_DTA_dataset(dataset)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False,  collate_fn=collate)
    # training the model
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    model = modeling().to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    epochs = 2000
    best_mse = 1000
    best_epoch = -1

    for epoch in range(epochs):
        train(model, device, train_loader, optimizer, epoch+1)
        G,P = predicting(model, device, test_loader)
        test_loss =  mse(G,P)
        msg =  "epoch-%d, mse-%.4f"%(epoch+1,test_loss)
        logger.info(msg)
        if test_loss<best_mse:
            save_model_dict(model, logger.get_model_dir(), msg)
            best_epoch = epoch+1
            best_mse = test_loss
            print('rmse improved at epoch ', best_epoch, '; best_mse:', best_mse,model_st,dataset)
        else:
            print('No improvement since epoch ', best_epoch, '; best_mse:', best_mse,model_st,dataset)
    print('train success!')
