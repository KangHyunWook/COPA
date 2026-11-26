from sklearn.metrics import accuracy_score, f1_score
from data_loader import get_loader
from config import get_config
from solver import Solver
from random import random

import numpy as np
import models
import torch
import time

start_subject=1
def run(train_config):
    test_acc_list=[]
    test_f1_list =[]
    
    elapsed_time=0
    for i in range(start_subject, train_config.n_subjects):
        train_config.name=f'sub{i}'
        dev_config = get_config(mode='dev')
        test_config = get_config(mode='test')

        train_config.subject=dev_config.subject=test_config.subject=str(i)
        train_data_loader = get_loader(train_config, shuffle=True)
        dev_data_loader = get_loader(dev_config, shuffle=False)
        test_data_loader = get_loader(test_config, shuffle=False)
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        solver = Solver
        solver = solver(train_config, dev_config, test_config, train_data_loader, dev_data_loader, test_data_loader, device, is_train=True, model=None)
        
        solver.build()
        start = time.perf_counter() 
        solver.train()
        end=time.perf_counter()
        elapsed_time += (end-start)

        test_loss, test_true, test_pred = solver.eval(mode="test", to_print=True)

        test_acc=accuracy_score(test_true, test_pred)
        test_f1 = f1_score(test_true, test_pred, average = 'weighted')

        test_acc_list.append(test_acc)
        test_f1_list.append(test_f1)
        
    training_time=elapsed_time/15
    print('avg training time:', training_time)

    avg_cls_acc=np.round(np.mean(test_acc_list)*100,3)    
    avg_acc_std = np.round(np.std(test_acc_list)*100,3)
    avg_f1 = np.round(np.mean(test_f1_list)*100,3)
    avg_f1_std=np.round(np.std(test_f1_list)*100,3)
    
    print(f'{train_config.model_name} avg cls acc: {avg_cls_acc} / {avg_acc_std} | avg F1: {avg_f1} / {avg_f1_std}')


if __name__=='__main__':
    random_name = str(random())
    random_seed = 123 
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)

    train_config = get_config(mode='train')
    
    run(train_config)
