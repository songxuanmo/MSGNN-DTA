import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from dataloader import create_DTA_dataset
# from dataloader_test import create_DTA_dataset
from model import Graph_GAT, Graph_GCN, Graph_noSub, Graph_noConn, Graph_noATT
from utils import *

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

datasets = [['davis','kiba'][1]]
modeling = [Graph_GAT][0]
model_st = modeling.__name__
print('\npredicting for test dataset using ', model_st)
TEST_BATCH_SIZE = 512

result = []
_,test_data = create_DTA_dataset(datasets[0])
test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, collate_fn=collate)

# training the model
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
model = modeling().to(device)
model_file_name = 'models/KIBA_GAT_gmp.pt'
if os.path.isfile(model_file_name):
    model.load_state_dict(torch.load(model_file_name,map_location=torch.device('cpu')),strict=False)
    G,P = predicting(model, device, test_loader)
    ret = [mse(G, P), rmse(G, P), ci(G, P), r2s(G, P), pearson(G, P), spearman(G, P)]
    ret = [datasets[0],model_st]+[round(e,3) for e in ret]
    result += [ret]
    print('dataset,model,mse,rmse,ci,r2s,pearson,spearman')
    print(ret)
else:
    print('model is not available!')

with open('results/'+model_st+'_result_'+ datasets[0] +'.csv','a') as f:
    f.write('dataset,model,mse,rmse,ci,r2s,pearson,spearman\n')
    for ret in result:
        f.write(','.join(map(str,ret)) + '\n')


