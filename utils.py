import os
from torch_geometric.data import InMemoryDataset, DataLoader, Batch
from torch_geometric import data as DATA
import torch
from tqdm import tqdm
import numpy as np
from math import sqrt
from scipy import stats
from sklearn.metrics import r2_score

# initialize the dataset
class DTADataset(InMemoryDataset):
    def __init__(self, root='data', dataset='davis',
                 drug_smiles=None, target_sequence=None, x=None, x_mask=None, xt=None, xt_mask=None, y=None, transform=None,
                 pre_transform=None, smile_graph=None,  target_graph=None, clique_graph=None,):
        super(DTADataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.drug_smiles = drug_smiles
        self.target_sequence = target_sequence
        self.y = y
        self.smile_graph = smile_graph
        self.target_graph = target_graph
        self.clique_graph = clique_graph
        self.process(drug_smiles, target_sequence, x, x_mask, xt, xt_mask, y, smile_graph, target_graph, clique_graph)

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + '_data_mol.pt', self.dataset + '_data_pro.pt',  self.dataset + '_data_clique.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, drug_smiles=None, target_sequence=None, x=None, x_mask=None, xt=None, xt_mask=None, y=None, smile_graph=None, target_graph=None, clique_graph=None,):
        assert (len(drug_smiles) == len(target_sequence) and len(drug_smiles) == len(y)), 'The three lists must have the same length!'
        data_list_mol = []
        data_list_pro = []
        data_list_clique = []
        data_len = len(drug_smiles)
        print('loading tensors ...')
        for i in tqdm(range(data_len)):
            print('Converting to graph: {}/{}'.format(i + 1, data_len))
            smiles = drug_smiles[i]
            seq = target_sequence[i]
            if x is not None:
                drug = x[i]
            if x_mask is not None:
                drug_mask = x_mask[i]
            if xt is not None:
                target = xt[i]
            if xt_mask is not None:
                target_mask = xt_mask[i]
            labels = y[i]
            mol_size, mol_features, mol_edge_index = smile_graph[smiles]
            target_size, target_features, target_edge_index, target_edge_weight = target_graph[seq]
            clique_size, clique_features, clique_edge_index = clique_graph[smiles]

            GCNData_mol = DATA.Data(x=torch.Tensor(mol_features),
                                    edge_index=torch.LongTensor(mol_edge_index).transpose(1, 0),
                                    y=torch.FloatTensor([labels]))
            if x is not None:
                GCNData_mol.drug = torch.LongTensor([drug])
            if x_mask is not None:
                GCNData_mol.drug_mask = torch.LongTensor([drug_mask])
            GCNData_mol.__setitem__('c_size', torch.LongTensor([mol_size]))

            GCNData_pro = DATA.Data(x=torch.Tensor(target_features),
                                    edge_index=torch.LongTensor(target_edge_index).transpose(1, 0),
                                    edge_weight=torch.FloatTensor(target_edge_weight),
                                    y=torch.FloatTensor([labels]))
            if xt is not None:
                GCNData_pro.target = torch.LongTensor([target])
            if xt_mask is not None:
                GCNData_pro.target_mask = torch.LongTensor([target_mask])
            GCNData_pro.__setitem__('target_size', torch.LongTensor([target_size]))

            GCNData_clique = DATA.Data(x=torch.Tensor(clique_features),
                                    edge_index=torch.LongTensor(clique_edge_index).transpose(1, 0),
                                    y=torch.FloatTensor([labels]))
            GCNData_clique.__setitem__('target_size', torch.LongTensor([clique_size]))

            data_list_mol.append(GCNData_mol)
            data_list_pro.append(GCNData_pro)
            data_list_clique.append(GCNData_clique)

        if self.pre_filter is not None:
            data_list_mol = [data for data in data_list_mol if self.pre_filter(data)]
            data_list_pro = [data for data in data_list_pro if self.pre_filter(data)]
            data_list_clique = [data for data in data_list_clique if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list_mol = [self.pre_transform(data) for data in data_list_mol]
            data_list_pro = [self.pre_transform(data) for data in data_list_pro]
            data_list_clique = [self.pre_transform(data) for data in data_list_clique]

        self.data_mol = data_list_mol
        self.data_pro = data_list_pro
        self.data_clique = data_list_clique


    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # return GNNData_mol, GNNData_pro
        return self.data_mol[idx], self.data_pro[idx], self.data_clique[idx]

def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse
def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse
def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp
def spearman(y,f):
    rs = stats.spearmanr(y, f)[0]
    return rs
def ci(y,f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci
def get_cindex(gt, pred):
    gt_mask = gt.reshape((1, -1)) > gt.reshape((-1, 1))
    diff = pred.reshape((1, -1)) - pred.reshape((-1, 1))
    h_one = (diff > 0)
    h_half = (diff == 0)
    CI = np.sum(gt_mask * h_one * 1.0 + gt_mask * h_half * 0.5) / np.sum(gt_mask)

    return CI

def r2s(y,f):
    r2s = r2_score(y, f)
    return r2s
class BestMeter(object):
    """Computes and stores the best value"""
    def __init__(self, best_type):
        self.best_type = best_type
        self.count = 0
        self.reset()

    def reset(self):
        if self.best_type == 'min':
            self.best = float('inf')
        else:
            self.best = -float('inf')

    def update(self, best):
        self.best = best
        self.count = 0

    def get_best(self):
        return self.best

    def counter(self):
        self.count += 1
        return self.count

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def get_average(self):
        self.avg = self.sum / (self.count + 1e-12)

        return self.avg

def save_model_dict(model, model_dir, msg):
    model_path = os.path.join(model_dir, msg + '.pt')
    torch.save(model.state_dict(), model_path)
    print("model has been saved to %s." % (model_path))

def load_model_dict(model, ckpt):
    model.load_state_dict(torch.load(ckpt))

def collate(data_list):
    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = Batch.from_data_list([data[1] for data in data_list])
    batchC = Batch.from_data_list([data[2] for data in data_list])
    return batchA, batchB, batchC
