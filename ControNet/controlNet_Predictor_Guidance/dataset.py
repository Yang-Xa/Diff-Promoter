import torch
from itertools import islice
from torch.utils import data
import random
random.seed(12301)

dict_promoter = {}
with open(r'../data/maize_seq.csv', 'r') as f:
    for line in islice(f, 1, None):
        tmp = line[:-1].split(',')
        dict_promoter[tmp[0]] = tmp[11]

list_train = []
with open(r'../data/train_promoter.txt', 'r') as f:
    for line in f:
        list_train.append(line[:-1])

list_val = []
with open(r'../data/val_promoter.txt', 'r') as f:
    for line in f:
        list_val.append(line[:-1])


class myDataset(data.Dataset):
    def __init__(self, list_name, dict_seq):
        self.list_name = list_name
        self.dict_seq = dict_seq

    def __getitem__(self, index):
        gene_seq = self.dict_seq[self.list_name[index]]

        one_hot = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [0, 0, 0, 0], 
                    'a': [1, 0, 0, 0], 'c': [0, 1, 0, 0], 'g': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'n': [0, 0, 0, 0]}

        encode_list = []
        for element in 'NNN' + gene_seq + 'NNN':
            encode_list.append(one_hot[element])
        seq = torch.tensor(encode_list, dtype=torch.float).t()

        idx = random.randint(0, 164)
        cond_list = []
        for element in 'NNN' + gene_seq[:idx] + 'N' * 6 + gene_seq[idx+6: ] + 'NNN':
            cond_list.append(one_hot[element])
        cond_seq = torch.tensor(cond_list, dtype=torch.float).t()

        return seq, cond_seq

    def __len__(self):
        return len(self.list_name)


# dataset_train = myDataset(list_train + list_val, dict_promoter)
# dataset_test = myDataset(list_test, dict_promoter)
dataset_val = myDataset(list_val[:3000]+list_train[-5000:], dict_promoter)

# train_loader = data.DataLoader(
#     dataset=dataset_train,
#     batch_size=64,
#     shuffle=True
# )

val_loader = data.DataLoader(
    dataset=dataset_val,
    batch_size=64,
    shuffle=True
)


# test_loader = data.DataLoader(
#     dataset=dataset_test,
#     batch_size=64,
#     shuffle=True
# )


if __name__ == '__main__':
    for seq, cond_seq in val_loader:
        print(seq.shape, cond_seq.shape)

