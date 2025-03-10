import torch
from itertools import islice
from torch.utils import data


dict_promoter = {}
with open(r'./data/maize_seq.csv', 'r') as f:
    for line in islice(f, 1, None):
        tmp = line[:-1].split(',')
        dict_promoter[tmp[0]] = tmp[11]


list_train, list_val, list_test = [], [], []
with open(r'./data/train_promoter.txt', 'r') as f:
    for line in f:
        list_train.append(line[:-1])

with open(r'./data/val_promoter.txt', 'r') as f:
    for line in f:
        list_val.append(line[:-1])

with open(r'./data/test_promoter.txt', 'r') as f:
    for line in f:
        list_test.append(line[:-1])


class myDataset(data.Dataset):
    def __init__(self, list_name, dict_seq):
        self.list_name = list_name
        self.dict_seq = dict_seq

    def __getitem__(self, index):
        gene_seq = self.dict_seq[self.list_name[index]]

        one_hot = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [0, 0, 0, 0], 
                    'a': [1, 0, 0, 0], 'c': [0, 1, 0, 0], 'g': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'n': [0, 0, 0, 0]}

        encode_list = []
        for element in gene_seq:
            encode_list.append(one_hot[element])

        seq = torch.tensor(encode_list, dtype=torch.float).t()

        return seq

    def __len__(self):
        return len(self.list_name)


dataset_train = myDataset(list_train + list_val, dict_promoter)
dataset_test = myDataset(list_test, dict_promoter)

train_loader = data.DataLoader(
    dataset=dataset_train,
    batch_size=64,
    shuffle=True,
    drop_last=True
)

test_loader = data.DataLoader(
    dataset=dataset_test,
    batch_size=64,
    shuffle=True,
    drop_last=True
)


if __name__ == '__main__':
    for seq in train_loader:
        print(seq.shape)
        break
    print(len(list_train) + len(list_val))