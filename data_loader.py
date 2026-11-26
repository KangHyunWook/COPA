from torch.utils.data import Dataset, DataLoader
from create_dataset import SEED, SEED_IV

import torch
import numpy as np

class EEGDataset(Dataset):
    def __init__(self, config):
        if 'seed_iv' in str(config.dataset_dir).lower():
            dataset=SEED_IV(config)
        elif 'seed' in str(config.dataset_dir).lower():
            dataset=SEED(config)
        else:
            print('Dataset not defined correctly')
            exit()
            
        self.data = dataset.get_data(config.mode)
        self.len = len(self.data)
    def __getitem__(self, index):
        return self.data[index]  
    
    def __len__(self):
        return self.len  
        
def get_loader(config, shuffle=True):
    dataset = EEGDataset(config)

    def collate_fn(batch):
        if config.label_type=='discrete' or config.label_type=='valence':
            labels = torch.cat([torch.from_numpy(np.asarray([sample[1]])) for sample in batch])   
        elif config.label_type=='arousal':
            labels = torch.cat([torch.from_numpy(np.asarray([sample[2]])) for sample in batch])
        else:
            print('wrong label type')
            exit(1)

        features = torch.cat([torch.FloatTensor(np.asarray([sample[0]])) for sample in batch])
        return features, labels
        
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn)
        
    return data_loader
