import torch
from torch.utils.data import Dataset


class LoaderH5(Dataset):

    def __init__(self, data):

        self.x = data['x']
        self.y = data['y'] - 1  #Class IDs start from 1 in the h5 files
        self.a = data['a']

    def __getitem__(self, index):
        return {
            'x': torch.from_numpy(self.x[index]).type(torch.FloatTensor),
            'y': int(self.y[index]),
            'a': torch.from_numpy(self.a[index]).type(torch.FloatTensor)
        }

    def __len__(self):
        return self.y.shape[0]
