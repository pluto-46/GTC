#!/usr/bin/env python
# coding: utf-8

# ## Import necessary libraries


# #### Check if CUDA is available

# In[1]:


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# ## Helper functions


# dictionary of atoms where a new element gets a new index
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
    adjacency = Chem.GetAdjacencyMatrix(mol)
    n = adjacency.shape[0]

    adjacency = adjacency + np.eye(n)
    degree = sum(adjacency)
    d_half = np.sqrt(np.diag(degree))
    d_half_inv = np.linalg.inv(d_half)
    adjacency = np.matmul(d_half_inv, np.matmul(adjacency, d_half_inv))
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


# ## Data processing

# In[4]:


radius = 2

with open('kegg_classes.txt', 'r') as f:
    data_list = f.read().strip().split('\n')

"""Exclude the data contains "." in the smiles, which correspond to non-bonds"""
data_list = list(filter(lambda x: '.' not in x.strip().split()[0], data_list))
N = len(data_list)

print('Total number of molecules : %d' % (N))

atom_dict = defaultdict(lambda: len(atom_dict))
bond_dict = defaultdict(lambda: len(bond_dict))
fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))

Molecules, Adjacencies, Properties, MACCS_list = [], [], [], []

max_MolMR, min_MolMR = -1000, 1000
max_MolLogP, min_MolLogP = -1000, 1000
max_MolWt, min_MolWt = -1000, 1000
max_NumRotatableBonds, min_NumRotatableBonds = -1000, 1000
max_NumAliphaticRings, min_NumAliphaticRings = -1000, 1000
max_NumAromaticRings, min_NumAromaticRings = -1000, 1000
max_NumSaturatedRings, min_NumSaturatedRings = -1000, 1000

for no, data in enumerate(data_list):

    print('/'.join(map(str, [no + 1, N])))

    smiles, property_indices = data.strip().split('\t')
    property_s = property_indices.strip().split(',')

    property = np.zeros((1, 11))
    for prop in property_s:
        property[0, int(prop)] = 1

    Properties.append(property)

    mol = Chem.MolFromSmiles(smiles)
    atoms = create_atoms(mol)
    i_jbond_dict = create_ijbonddict(mol)

    fingerprints = create_fingerprints(atoms, i_jbond_dict, radius)
    Molecules.append(fingerprints)

    adjacency = create_adjacency(mol)
    Adjacencies.append(adjacency)

    MACCS = MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(smiles))
    MACCS_ids = np.zeros((20,))
    MACCS_ids[0] = Descriptors.MolMR(mol)
    MACCS_ids[1] = Descriptors.MolLogP(mol)
    MACCS_ids[2] = Descriptors.MolWt(mol)
    MACCS_ids[3] = Descriptors.NumRotatableBonds(mol)
    MACCS_ids[4] = Descriptors.NumAliphaticRings(mol)
    MACCS_ids[5] = MACCS[108]
    MACCS_ids[6] = Descriptors.NumAromaticRings(mol)
    MACCS_ids[7] = MACCS[98]
    MACCS_ids[8] = Descriptors.NumSaturatedRings(mol)
    MACCS_ids[9] = MACCS[137]
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

dir_input = ('./pathway/input' + str(radius) + '/')
os.makedirs(dir_input, exist_ok=True)

for n in range(N):
    for b in range(20):
        if b == 0:
            MACCS_list[n][b] = (MACCS_list[n][b] - min_MolMR) / (max_MolMR - min_MolMR)
        elif b == 1:
            MACCS_list[n][b] = (MACCS_list[n][b] - min_MolLogP) / (max_MolMR - min_MolLogP)
        elif b == 2:
            MACCS_list[n][b] = (MACCS_list[n][b] - min_MolWt) / (max_MolMR - min_MolWt)
        elif b == 3:
            MACCS_list[n][b] = (MACCS_list[n][b] - min_NumRotatableBonds) / (max_MolMR - min_NumRotatableBonds)
        elif b == 4:
            MACCS_list[n][b] = (MACCS_list[n][b] - min_NumAliphaticRings) / (max_MolMR - min_NumAliphaticRings)
        elif b == 6:
            MACCS_list[n][b] = (MACCS_list[n][b] - min_NumAromaticRings) / (max_MolMR - min_NumAromaticRings)
        elif b == 8:
            MACCS_list[n][b] = (MACCS_list[n][b] - min_NumSaturatedRings) / (
                        max_NumSaturatedRings - min_NumSaturatedRings)

# np.save(dir_input + 'molecules', Molecules)
with open(dir_input + 'molecules.pkl', 'wb') as f:
    pickle.dump(Molecules, f)
# np.save(dir_input + 'adjacencies', Adjacencies)
with open(dir_input + 'adjacencies.pkl', 'wb') as f:
    pickle.dump(Adjacencies, f)
np.save(dir_input + 'properties', Properties)
np.save(dir_input + 'maccs', np.asarray(MACCS_list))



# with open(dir_input + 'properties.pkl', 'wb') as f:
#     pickle.dump(Properties, f)
#
# with open(dir_input + 'maccs.pkl', 'wb') as f:
#     pickle.dump(MACCS_list, f)


dump_dictionary(fingerprint_dict, dir_input + 'fingerprint_dict.pickle')

print('The preprocess has finished!')


# ## Define GNN

# In[5]:


class PathwayPredictor(nn.Module):

    def __init__(self):
        super(PathwayPredictor, self).__init__()
        self.embed_atom = nn.Embedding(n_fingerprint, dim)
        self.W_atom = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layer)])
        self.W_property = nn.Linear(dim + extra_dim, 11)

    """Pad adjacency matrices for batch processing."""

    def pad(self, matrices, value):
        sizes = [d.shape[0] for d in matrices]
        D = sum(sizes)
        pad_matrices = value + np.zeros((D, D))
        m = 0
        for i, d in enumerate(matrices):
            s_i = sizes[i]
            pad_matrices[m:m + s_i, m:m + s_i] = d
            m += s_i
        return torch.FloatTensor(pad_matrices).to(device)

    def sum_axis(self, xs, axis):
        y = list(map(lambda x: torch.sum(x, 0), torch.split(xs, axis)))
        return torch.stack(y)

    def update(self, xs, adjacency, i):
        hs = torch.relu(self.W_atom[i](xs))
        return torch.matmul(adjacency, hs)

    def forward(self, inputs, sel_maccs):

        atoms, adjacency = inputs

        axis = list(map(lambda x: len(x), atoms))

        atoms = torch.cat(atoms)

        x_atoms = self.embed_atom(atoms)
        adjacency = self.pad(adjacency, 0)

        for i in range(layer):
            x_atoms = self.update(x_atoms, adjacency, i)

        extra_inputs = sel_maccs.to(device)
        y_molecules = self.sum_axis(x_atoms, axis)

        y_molecules = torch.cat((y_molecules, extra_inputs), 1)
        z_properties = self.W_property(y_molecules)

        return z_properties

    def __call__(self, data_batch, train=True):
        # sel_maccs = torch.FloatTensor(data_batch[-1])
        # #sel_maccs = torch.tensor(data_batch[-1], dtype=torch.float)
        #
        # inputs, t_properties = data_batch[:-2], torch.cat(data_batch[-2])
        #
        # z_properties = self.forward(inputs, sel_maccs)
        # 修复代码
        sel_maccs = torch.stack(data_batch[-1]).float().to(device)  # 显式堆叠

        inputs, t_properties = data_batch[:-2], torch.cat(data_batch[-2])
        z_properties = self.forward(inputs, sel_maccs)

        if train:
            loss = F.binary_cross_entropy(torch.sigmoid(z_properties), t_properties)
            return loss
        else:
            zs = torch.sigmoid(z_properties).to('cpu').data.numpy()
            ts = t_properties.to('cpu').data.numpy()
            scores = list(map(lambda x: x, zs))
            labels = list(map(lambda x: (x >= 0.5).astype(int), zs))
            return scores, labels, ts


# ## Train and test routines

# In[6]:


class Trainer(object):

    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, dataset_train):
        np.random.shuffle(dataset_train)
        N = len(dataset_train)
        loss_total = 0
        for i in range(0, N, batch):
            data_batch = list(zip(*dataset_train[i:i + batch]))
            loss = self.model(data_batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.to('cpu').data.numpy()
        return loss_total


class Tester(object):

    def __init__(self, model):
        self.model = model

    def test(self, dataset_test):
        N = len(dataset_test)
        score_list, label_list, t_list = [], [], []

        for i in range(0, N, batch):
            data_batch = list(zip(*dataset_test[i:i + batch]))
            scores, labels, ts = self.model(data_batch, train=False)
            score_list = np.append(score_list, scores)
            label_list = np.append(label_list, labels)
            t_list = np.append(t_list, ts)

        auc = accuracy_score(t_list, label_list)
        precision = precision_score(t_list, label_list)
        recall = recall_score(t_list, label_list)

        return auc, precision, recall


# ## Define GNN parameters

# In[7]:


dim = 50
layer = 2
batch = 10
lr = 1e-3
lr_decay = 0.75
decay_interval = 20
iteration = 100
extra_dim = 20

(dim, layer, batch, decay_interval, iteration, extra_dim) = map(int, [dim, layer, batch, decay_interval, iteration,
                                                                      extra_dim])
lr, lr_decay = map(float, [lr, lr_decay])

# ## Load and split data

# In[8]:


dir_input = ('pathway/input' + str(radius) + '/')

# molecules = load_tensor(dir_input + 'molecules', torch.LongTensor)
# adjacencies = load_numpy(dir_input + 'adjacencies')
# t_properties = load_tensor(dir_input + 'properties', torch.FloatTensor)
# maccs = load_numpy(dir_input + 'maccs')

# 加载 molecules.pkl，并转换为 torch.LongTensor
with open(dir_input + 'molecules.pkl', 'rb') as f:
    molecules_data = pickle.load(f)
molecules = [torch.LongTensor(item) for item in molecules_data]

# 加载 adjacencies.pkl（一般为 list 或 ndarray，不用转换为 tensor）
with open(dir_input + 'adjacencies.pkl', 'rb') as f:
    adjacencies = pickle.load(f)

# 加载 properties.pkl，并转换为 torch.FloatTensor
with open(dir_input + 'properties.pkl', 'rb') as f:
    properties_data = pickle.load(f)
t_properties = torch.FloatTensor(properties_data)

# 加载 maccs.pkl（视需求可转换为 tensor）
with open(dir_input + 'maccs.pkl', 'rb') as f:
    maccs_data = pickle.load(f)
# 若需要转换为 tensor（如用于模型输入），则使用：
maccs = torch.FloatTensor(maccs_data)


with open(dir_input + 'fingerprint_dict.pickle', 'rb') as f:
    fingerprint_dict = pickle.load(f)

dataset = list(zip(molecules, adjacencies, t_properties, maccs))
dataset = shuffle_dataset(dataset, 1234)
dataset_train, dataset_ = split_dataset(dataset, 0.8)
dataset_dev, dataset_test = split_dataset(dataset_, 0.5)

fingerprint_dict = load_pickle(dir_input + 'fingerprint_dict.pickle')
unknown = 100
n_fingerprint = len(fingerprint_dict) + unknown

# ## Begin Training

# In[9]:


torch.manual_seed(1234)

model = PathwayPredictor().to(device)
trainer = Trainer(model)
tester = Tester(model)

dir_output = ('pathway/output/')
os.makedirs(dir_output, exist_ok=True)

print('Training...')
print('Epoch \t Time(sec) \t Loss_train \t AUC_dev \t AUC_test \t Precision \t Recall')

start = timeit.default_timer()

for epoch in range(iteration):
    if (epoch + 1) % decay_interval == 0:
        trainer.optimizer.param_groups[0]['lr'] *= lr_decay

    loss = trainer.train(dataset_train)
    auc_dev = tester.test(dataset_dev)[0]
    auc_test, precision, recall = tester.test(dataset_test)

    lr_rate = trainer.optimizer.param_groups[0]['lr']

    end = timeit.default_timer()
    time = end - start

    print('%d \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f' % (
    epoch, time, loss, auc_dev, auc_test, precision, recall))

# #### Random evaluation

# In[10]:


data_batch = list(zip(*dataset_test[0:0 + batch]))

# sel_maccs = torch.FloatTensor(data_batch[-1])
# inputs, t_properties = data_batch[:-2], torch.cat(data_batch[-2])
# z_properties = model.forward(inputs, sel_maccs)

# 修复代码
sel_maccs = torch.stack(data_batch[-1]).float().to(device)  # 显式堆叠

inputs, t_properties = data_batch[:-2], torch.cat(data_batch[-2])
z_properties = model.forward(inputs, sel_maccs)

# True classes
print(t_properties)

# Predicted classes
torch.set_printoptions(precision=2)
p_properties = torch.sigmoid(z_properties)

for j in range(batch):
    print('%.2f\b %.2f\b %.2f\b %.2f\b %.2f\b %.2f\b %.2f\b %.2f\b %.2f\b %.2f\b %.2f\n' % (p_properties[j, 0], \
                                                                                            p_properties[j, 1],
                                                                                            p_properties[j, 2],
                                                                                            p_properties[j, 3],
                                                                                            p_properties[j, 4],
                                                                                            p_properties[j, 5],
                                                                                            p_properties[j, 6], \
                                                                                            p_properties[j, 7],
                                                                                            p_properties[j, 8],
                                                                                            p_properties[j, 9],
                                                                                            p_properties[j, 10]))

# ### Class-wise statistics

# In[11]:


data_batch = list(zip(*dataset_test[:]))

# sel_maccs = torch.FloatTensor(data_batch[-1])
# inputs, t_properties = data_batch[:-2], torch.cat(data_batch[-2])
# z_properties = model.forward(inputs, sel_maccs)

sel_maccs = torch.stack(data_batch[-1]).float().to(device)  # 显式堆叠

inputs, t_properties = data_batch[:-2], torch.cat(data_batch[-2])
z_properties = model.forward(inputs, sel_maccs)

torch.set_printoptions(precision=2)
p_properties = torch.sigmoid(z_properties)

p_properties = p_properties.data.to('cpu').numpy()
t_properties = t_properties.data.to('cpu').numpy()

p_properties[p_properties < 0.5] = 0
p_properties[p_properties >= 0.5] = 1

for c in range(11):
    y_true = t_properties[:, c]
    y_pred = p_properties[:, c]

    auc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    print('Class ' + str(c + 1) + ' statistics:')
    print('Accuracy %.4f, Precision %.4f, Recall %.4f\n' % (auc, precision, recall))

