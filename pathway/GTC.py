
#!/usr/bin/env python
# coding: utf-8

# ## Import necessary libraries

# In[1]:


from collections import defaultdict
import os
import pickle
import sys
import timeit

import numpy as np

from rdkit import Chem
from rdkit.Chem import rdDepictor, Descriptors
from rdkit.Chem import MACCSkeys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, auc, roc_curve

from sklearn.exceptions import UndefinedMetricWarning

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


# #### Check if CUDA is available

# In[2]:


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')



# In[3]:


def create_atoms(mol):
    atoms = [atom_dict[a.GetSymbol()] for a in mol.GetAtoms()]
    return np.array(atoms)

# format from_atomIDx : [to_atomIDx, bondDict]
def create_ijbonddict(mol):
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict


def create_fingerprints(atoms, i_jbond_dict, radius):
    """Extract the r-radius subgraphs (i.e., fingerprints)
    from a molecular graph using WeisfeilerLehman-like algorithm."""

    if (len(atoms) == 1) or (radius == 0):
        fingerprints = [fingerprint_dict[a] for a in atoms]

    else:
        vertices = atoms
        for _ in range(radius):
            fingerprints = []
            for i, j_bond in i_jbond_dict.items():
                neighbors = [(vertices[j], bond) for j, bond in j_bond]
                fingerprint = (vertices[i], tuple(sorted(neighbors)))
                fingerprints.append(fingerprint_dict[fingerprint])
            vertices = fingerprints

    return np.array(fingerprints)


def create_adjacency(mol):
    adjacency  = Chem.GetAdjacencyMatrix(mol)
    n          = adjacency.shape[0]

    adjacency  = adjacency + np.eye(n)
    degree     = sum(adjacency)
    d_half     = np.sqrt(np.diag(degree))
    d_half_inv = np.linalg.inv(d_half)
    adjacency  = np.matmul(d_half_inv,np.matmul(adjacency,d_half_inv))
    return np.array(adjacency)


def dump_dictionary(dictionary, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(dict(dictionary), f)


def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy', allow_pickle=True)]


def load_numpy(file_name):
    return np.load(file_name + '.npy', allow_pickle=True)


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(MLPClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# ## Data processing

# In[4]:


radius = 2

with open('kegg_classes.txt', 'r') as f:
    data_list = f.read().strip().split('\n')

"""Exclude the data contains "." in the smiles, which correspond to non-bonds"""
data_list = list(filter(lambda x: '.' not in x.strip().split()[0], data_list))
N = len(data_list)

print('Total number of molecules : %d' %(N))

atom_dict = defaultdict(lambda: len(atom_dict))
bond_dict = defaultdict(lambda: len(bond_dict))
fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))

Molecules, Adjacencies, Properties, MACCS_list = [], [], [], []

max_MolMR, min_MolMR     = -1000, 1000
max_MolLogP, min_MolLogP = -1000, 1000
max_MolWt, min_MolWt     = -1000, 1000
max_NumRotatableBonds, min_NumRotatableBonds = -1000, 1000
max_NumAliphaticRings, min_NumAliphaticRings = -1000, 1000
max_NumAromaticRings, min_NumAromaticRings   = -1000, 1000
max_NumSaturatedRings, min_NumSaturatedRings = -1000, 1000

for no, data in enumerate(data_list):

    print('/'.join(map(str, [no+1, N])))

    smiles, property_indices = data.strip().split('\t')
    property_s = property_indices.strip().split(',')

    property = np.zeros((1,11))
    for prop in property_s:
        property[0,int(prop)] = 1

    Properties.append(property)

    mol = Chem.MolFromSmiles(smiles)
    atoms = create_atoms(mol)
    i_jbond_dict = create_ijbonddict(mol)

    fingerprints = create_fingerprints(atoms, i_jbond_dict, radius)
    Molecules.append(fingerprints)

    adjacency = create_adjacency(mol)
    Adjacencies.append(adjacency)

    MACCS         = MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(smiles))
    MACCS_ids     = np.zeros((20,))
    MACCS_ids[0]  = Descriptors.MolMR(mol)
    MACCS_ids[1]  = Descriptors.MolLogP(mol)
    MACCS_ids[2]  = Descriptors.MolWt(mol)
    MACCS_ids[3]  = Descriptors.NumRotatableBonds(mol)
    MACCS_ids[4]  = Descriptors.NumAliphaticRings(mol)
    MACCS_ids[5]  = MACCS[108]
    MACCS_ids[6]  = Descriptors.NumAromaticRings(mol)
    MACCS_ids[7]  = MACCS[98]
    MACCS_ids[8]  = Descriptors.NumSaturatedRings(mol)
    MACCS_ids[9]  = MACCS[137]
    MACCS_ids[10] = MACCS[136]
    MACCS_ids[11] = MACCS[145]
    MACCS_ids[12] = MACCS[116]
    MACCS_ids[13] = MACCS[141]
    MACCS_ids[14] = MACCS[89]
    MACCS_ids[15] = MACCS[50]
    MACCS_ids[16] = MACCS[160]
    MACCS_ids[17] = MACCS[121]
    MACCS_ids[18] = MACCS[149]
    MACCS_ids[19] = MACCS[161]

    if max_MolMR < MACCS_ids[0]:
        max_MolMR = MACCS_ids[0]
    if min_MolMR > MACCS_ids[0]:
        min_MolMR = MACCS_ids[0]

    if max_MolLogP < MACCS_ids[1]:
        max_MolLogP = MACCS_ids[1]
    if min_MolLogP > MACCS_ids[1]:
        min_MolLogP = MACCS_ids[1]

    if max_MolWt < MACCS_ids[2]:
        max_MolWt = MACCS_ids[2]
    if min_MolWt > MACCS_ids[2]:
        min_MolWt = MACCS_ids[2]

    if max_NumRotatableBonds < MACCS_ids[3]:
        max_NumRotatableBonds = MACCS_ids[3]
    if min_NumRotatableBonds > MACCS_ids[3]:
        min_NumRotatableBonds = MACCS_ids[3]

    if max_NumAliphaticRings < MACCS_ids[4]:
        max_NumAliphaticRings = MACCS_ids[4]
    if min_NumAliphaticRings > MACCS_ids[4]:
        min_NumAliphaticRings = MACCS_ids[4]

    if max_NumAromaticRings < MACCS_ids[6]:
        max_NumAromaticRings = MACCS_ids[6]
    if min_NumAromaticRings > MACCS_ids[6]:
        min_NumAromaticRings = MACCS_ids[6]

    if max_NumSaturatedRings < MACCS_ids[8]:
        max_NumSaturatedRings = MACCS_ids[8]
    if min_NumSaturatedRings > MACCS_ids[8]:
        min_NumSaturatedRings = MACCS_ids[8]

    MACCS_list.append(MACCS_ids)

dir_input = ('pathway/input'+str(radius)+'/')
os.makedirs(dir_input, exist_ok=True)

for n in range(N):
    for b in range(20):
        if b==0:
            MACCS_list[n][b] = (MACCS_list[n][b]-min_MolMR)/(max_MolMR-min_MolMR)
        elif b==1:
            MACCS_list[n][b] = (MACCS_list[n][b]-min_MolLogP)/(max_MolMR-min_MolLogP)
        elif b==2:
            MACCS_list[n][b] = (MACCS_list[n][b]-min_MolWt)/(max_MolMR-min_MolWt)
        elif b==3:
            MACCS_list[n][b] = (MACCS_list[n][b]-min_NumRotatableBonds)/(max_MolMR-min_NumRotatableBonds)
        elif b==4:
            MACCS_list[n][b] = (MACCS_list[n][b]-min_NumAliphaticRings)/(max_MolMR-min_NumAliphaticRings)
        elif b==6:
            MACCS_list[n][b] = (MACCS_list[n][b]-min_NumAromaticRings)/(max_MolMR-min_NumAromaticRings)
        elif b==8:
            MACCS_list[n][b] = (MACCS_list[n][b]-min_NumSaturatedRings)/(max_NumSaturatedRings-min_NumSaturatedRings)

# np.save(dir_input + 'molecules', Molecules)
with open(dir_input + 'molecules.pkl', 'wb') as f:
    pickle.dump(Molecules, f)
# np.save(dir_input + 'adjacencies', Adjacencies)
with open(dir_input + 'adjacencies.pkl', 'wb') as f:
    pickle.dump(Adjacencies, f)
np.save(dir_input + 'properties', Properties)
np.save(dir_input + 'maccs', np.asarray(MACCS_list))

dump_dictionary(fingerprint_dict, dir_input + 'fingerprint_dict.pickle')

print('The preprocess has finished!')


# ## Load and create dataset

# In[5]:


dir_input = ('pathway/input'+str(radius)+'/')

# molecules  = load_tensor(dir_input + 'molecules', torch.FloatTensor)
# properties = load_numpy(dir_input + 'properties')
# maccs      = load_numpy(dir_input + 'maccs')

# 加载 molecules.pkl，并转换为 torch.LongTensor
with open(dir_input + 'molecules.pkl', 'rb') as f:
    molecules_data = pickle.load(f)
molecules = [torch.LongTensor(item) for item in molecules_data]

# 加载 properties.pkl，并转换为 torch.FloatTensor
with open(dir_input + 'properties.pkl', 'rb') as f:
    properties_data = pickle.load(f)
properties = torch.FloatTensor(properties_data)

# 加载 maccs.pkl（视需求可转换为 tensor）
with open(dir_input + 'maccs.pkl', 'rb') as f:
    maccs_data = pickle.load(f)
# 若需要转换为 tensor（如用于模型输入），则使用：
maccs = torch.FloatTensor(maccs_data)


with open(dir_input + 'fingerprint_dict.pickle', 'rb') as f:
    fingerprint_dict = pickle.load(f)

fingerprint_dict = load_pickle(dir_input + 'fingerprint_dict.pickle')
unknown          = 100
n_fingerprint    = len(fingerprint_dict) + unknown

my_maccs = []
for i in range(len(molecules)):
    target_mol = (n_fingerprint-1)*torch.ones([259], dtype=torch.float, device=device)
    target_mol[:molecules[i].size()[0]] = molecules[i]
    my_maccs.append(np.concatenate((target_mol.cpu().data.numpy(),maccs[i]), axis=0))

dataset = list(zip(properties, my_maccs))
dataset = shuffle_dataset(dataset, 4123)
dataset_train, dataset_   = split_dataset(dataset, 0.8)
dataset_dev, dataset_test = split_dataset(dataset_, 0.5)


data_batch = list(zip(*dataset_train))
properties_train, maccs_train = data_batch[-2], data_batch[-1]

data_batch = list(zip(*dataset_dev))
properties_dev, maccs_dev = data_batch[-2], data_batch[-1]

data_batch = list(zip(*dataset_test))
properties_test, maccs_test = data_batch[-2], data_batch[-1]

train_len, dev_len, test_len = len(dataset_train), len(dataset_dev), len(dataset_test)

feature_len = maccs_train[0].shape[0]

X_train, X_dev, X_test = np.zeros((train_len,feature_len)), np.zeros((dev_len,feature_len)), np.zeros((test_len,feature_len))
Y_train, Y_dev, Y_test = np.zeros((train_len,11)), np.zeros((dev_len,11)), np.zeros((test_len,11))

for i in range(train_len):
    X_train[i,:] = maccs_train[i]
    Y_train[i] = properties_train[i][0]

for i in range(dev_len):
    X_dev[i,:]   = maccs_dev[i]
    Y_dev[i]   = properties_dev[i][0]

for i in range(test_len):
    X_test[i,:]  = maccs_test[i]
    Y_test[i]  = properties_test[i][0]



# In[6]:
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)

class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim=279, hidden_dim=128, output_dim=10, lr=1e-3, epochs=10, batch_size=64, device='cpu'):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device(device)

        self.model = MLP(input_dim, hidden_dim, output_dim).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()  # 多标签分类
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def fit(self, X, y):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).to(self.device)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            self.model.train()
            for xb, yb in loader:
                self.optimizer.zero_grad()
                preds = self.model(xb)
                loss = self.criterion(preds, yb)
                loss.backward()
                self.optimizer.step()
        return self

    def predict(self, X):
        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X)
            preds = torch.sigmoid(logits)
        return (preds > 0.5).int().cpu().numpy()

    def predict_proba(self, X):
        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X)
            probs = torch.sigmoid(logits)
        return probs.cpu().numpy()

n_labels = Y_train.shape[1]
print("标签数量（输出维度）为：", n_labels)
# clf = RandomForestClassifier(n_estimators=300, criterion = 'gini', max_depth=60, random_state=0)
# clf = TorchMLPClassifier(device='cpu', epochs=20, output_dim=11)
#
#
#
# multi_target_forest = MultiOutputClassifier(clf, n_jobs=-1)
# multi_target_forest.fit(X_train, Y_train)
clf = TorchMLPClassifier(hidden_dim=128, epochs=20, lr=0.001, device='cpu', output_dim=11)
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)




# In[7]:


#Y_pred = multi_target_forest.predict(X_test)

acc_score, prec_score, rec_score = 0., 0., 0.
for i in range(Y_test.shape[0]):
    acc_score  += accuracy_score(Y_test[i],Y_pred[i])
    prec_score += precision_score(Y_test[i],Y_pred[i])
    rec_score  += recall_score(Y_test[i],Y_pred[i])

acc_score  = acc_score/Y_test.shape[0]
prec_score = prec_score/Y_test.shape[0]
rec_score  = rec_score/Y_test.shape[0]

print('Accuracy : %.4f%%, \t Precision : %.4f%%, \t, Recall : %.4f%%' %(acc_score, prec_score, rec_score))


# In[8]:


Y_pred = clf.predict(X_dev)

acc_score, prec_score, rec_score = 0., 0., 0.
for i in range(Y_dev.shape[0]):
    acc_score  += accuracy_score(Y_dev[i],Y_pred[i])
    prec_score += precision_score(Y_dev[i],Y_pred[i])
    rec_score  += recall_score(Y_dev[i],Y_pred[i])

acc_score  = acc_score/Y_dev.shape[0]
prec_score = prec_score/Y_dev.shape[0]
rec_score  = rec_score/Y_dev.shape[0]

print('Accuracy : %.4f%%, \t Precision : %.4f%%, \t, Recall : %.4f%%' %(acc_score, prec_score, rec_score))


# # Logistic regression

# In[9]:


lm = linear_model.LogisticRegression()
multi_target_logistic = MultiOutputClassifier(lm, n_jobs=-1)
multi_target_logistic.fit(X_train, Y_train)


# In[10]:


Y_pred = multi_target_logistic.predict(X_test)

acc_score, prec_score, rec_score = 0., 0., 0.
for i in range(Y_test.shape[0]):
    acc_score  += accuracy_score(Y_test[i],Y_pred[i])
    prec_score += precision_score(Y_test[i],Y_pred[i])
    rec_score  += recall_score(Y_test[i],Y_pred[i])

acc_score  = acc_score/Y_test.shape[0]
prec_score = prec_score/Y_test.shape[0]
rec_score  = rec_score/Y_test.shape[0]

print('Accuracy : %.4f%%, \t Precision : %.4f%%, \t, Recall : %.4f%%' %(acc_score, prec_score, rec_score))


# In[11]:


Y_pred = multi_target_logistic.predict(X_dev)

acc_score, prec_score, rec_score = 0., 0., 0.
for i in range(Y_dev.shape[0]):
    acc_score  += accuracy_score(Y_dev[i],Y_pred[i])
    prec_score += precision_score(Y_dev[i],Y_pred[i])
    rec_score  += recall_score(Y_dev[i],Y_pred[i])

acc_score  = acc_score/Y_dev.shape[0]
prec_score = prec_score/Y_dev.shape[0]
rec_score  = rec_score/Y_dev.shape[0]

print('Accuracy : %.4f%%, \t Precision : %.4f%%, \t, Recall : %.4f%%' %(acc_score, prec_score, rec_score))


# # kNN classifier

# In[12]:


kNN = KNeighborsClassifier()
multi_target_kNN = MultiOutputClassifier(kNN, n_jobs=-1)
multi_target_kNN.fit(X_train, Y_train)


# In[13]:


Y_pred = multi_target_kNN.predict(X_test)

acc_score, prec_score, rec_score = 0., 0., 0.
for i in range(Y_test.shape[0]):
    acc_score  += accuracy_score(Y_test[i],Y_pred[i])
    prec_score += precision_score(Y_test[i],Y_pred[i])
    rec_score  += recall_score(Y_test[i],Y_pred[i])

acc_score  = acc_score/Y_test.shape[0]
prec_score = prec_score/Y_test.shape[0]
rec_score  = rec_score/Y_test.shape[0]

print('Accuracy : %.4f%%, \t Precision : %.4f%%, \t, Recall : %.4f%%' %(acc_score, prec_score, rec_score))


# In[14]:


Y_pred = multi_target_kNN.predict(X_dev)

acc_score, prec_score, rec_score = 0., 0., 0.
for i in range(Y_dev.shape[0]):
    acc_score  += accuracy_score(Y_dev[i],Y_pred[i])
    prec_score += precision_score(Y_dev[i],Y_pred[i])
    rec_score  += recall_score(Y_dev[i],Y_pred[i])

acc_score  = acc_score/Y_dev.shape[0]
prec_score = prec_score/Y_dev.shape[0]
rec_score  = rec_score/Y_dev.shape[0]

print('Accuracy : %.4f%%, \t Precision : %.4f%%, \t, Recall : %.4f%%' %(acc_score, prec_score, rec_score))


