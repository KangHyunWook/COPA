from utils import DiffLoss, MSE, CMD
from sklearn.metrics import accuracy_score

import torch
import models
import numpy as np
import os

import torch.nn as nn

class Solver(object):
    def __init__(self, train_config, dev_config, test_config, train_data_loader, dev_data_loader, test_data_loader, device, is_train=True, model=None):
        self.train_config = train_config
        self.train_data_loader = train_data_loader
        self.dev_data_loader = dev_data_loader
        self.test_data_loader = test_data_loader
        self.model=model
        self.is_train=is_train
        self.device = device
        
    def build(self, cuda=True):
        if self.model is None:
            self.model = getattr(models, self.train_config.model_name)(self.train_config)
        
        if torch.cuda.is_available() and cuda:
            self.model.to(self.device)
            
        if self.is_train:
            self.optimizer = self.train_config.optimizer(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.train_config.learning_rate)
                
    def train(self):
        curr_patience = patience = self.train_config.patience
        num_trials = 1
        
        self.criterion = criterion = nn.CrossEntropyLoss(reduction="mean")
        self.loss_diff = DiffLoss()
        self.loss_recon = MSE()
        self.loss_cmd = CMD()

        best_valid_loss = float('inf')
        
        train_losses = []
        valid_losses = []

        for e in range(self.train_config.num_epochs):
            self.model.train()
            train_loss = []
            
            for batch in self.train_data_loader:
                self.model.zero_grad()
                features, labels = batch
                labels = labels.type(torch.LongTensor)
                
                batch_size = features.size(0)
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                y_tilde = self.model(features)
                
                cls_loss = criterion(y_tilde, labels)

                if self.train_config.model_name=='COPA':
                    incon_loss = self.get_incon_loss()
                    recon_loss = self.get_recon_loss()
                    cmd_loss=self.get_cmd_loss()

                    invar_loss = cmd_loss
                    loss = cls_loss + \
                        self.train_config.incon_weight*incon_loss + \
                        self.train_config.invar_weight * invar_loss + \
                        self.train_config.recon_weight * recon_loss
                else:
                    loss=cls_loss
                
                loss.backward()

                self.optimizer.step()
                
                train_loss.append(loss.item())
            
            train_losses.append(train_loss)
            print(f"Training loss: {round(np.mean(train_loss), 4)}")

            valid_loss, valid_true, valid_pred = self.eval(mode="dev", to_print=False)
            
            print(f"Current patience: {curr_patience}, current trial: {num_trials}.")
            if valid_loss <= best_valid_loss:
                best_valid_loss = valid_loss
                print("Found new best model on dev set!")
                if not os.path.exists('checkpoints'): os.makedirs('checkpoints')
                
                torch.save(self.model.state_dict(), f'./checkpoints/model_{self.train_config.name}.std')
                torch.save(self.optimizer.state_dict(), f'./checkpoints/optim_{self.train_config.name}.std')
                curr_patience = patience
            else:
                curr_patience -= 1
                if curr_patience <= -1:
                    print("Running out of patience, loading previous best model.")
                    num_trials -= 1
                    curr_patience = patience
                    self.model.load_state_dict(torch.load(f'./checkpoints/model_{self.train_config.name}.std'))
                    self.optimizer.load_state_dict(torch.load(f'./checkpoints/optim_{self.train_config.name}.std'))
                    print(f"Current learning rate: {self.optimizer.state_dict()['param_groups'][0]['lr']}")

            if num_trials <= 0:
                print("Running out of patience, early stopping.")
                break
       
    def get_cmd_loss(self,):
        loss = self.loss_cmd(self.model.shared_con, self.model.shared_loc,5)
        loss += self.loss_cmd(self.model.shared_con, self.model.shared_seq, 5)
        loss += self.loss_cmd(self.model.shared_seq, self.model.shared_loc, 5)
        loss = loss/3.0

        return loss
        
    def get_incon_loss(self):
        shared_con = self.model.shared_con
        shared_loc = self.model.shared_loc
        shared_seq = self.model.shared_seq
        private_con = self.model.private_con
        private_loc = self.model.private_loc
        private_seq = self.model.private_seq
       
        loss = self.loss_diff(private_con, shared_con)
        loss += self.loss_diff(private_loc, shared_loc)
        loss += self.loss_diff(private_seq, shared_seq)

        loss += self.loss_diff(private_seq, private_con)
        loss += self.loss_diff(private_seq, private_loc)
        loss += self.loss_diff(private_con, private_loc)

        return loss
        
    def get_recon_loss(self, ):
        loss = self.loss_recon(self.model.con_recon, self.model.con)
        loss += self.loss_recon(self.model.loc_recon, self.model.loc)
        loss += self.loss_recon(self.model.seq_recon, self.model.seq)
        loss = loss/3.0
        return loss

    def eval(self,mode=None, to_print=False):
        assert(mode is not None)
        self.model.eval()

        y_true, y_pred = [], []
        eval_loss, eval_loss_diff = [], []

        if mode == "dev":
            dataloader = self.dev_data_loader
        elif mode == "test":
            dataloader = self.test_data_loader

            if to_print:
                self.model.load_state_dict(torch.load(
                    f'checkpoints/model_{self.train_config.name}.std'))

        valid_incon_losses=[]
        valid_invar_losses=[]
        valid_recon_losses=[]
        
        with torch.no_grad():
            for batch in dataloader:
                self.model.zero_grad()
                features, labels = batch
                labels = labels.type(torch.LongTensor)

                features = features.to(self.device)
                labels = labels.to(self.device)
                y_tilde = self.model(features)

                if self.train_config.model_name=='COPA':
                    incon_loss = self.get_incon_loss()
                    recon_loss = self.get_recon_loss()
                    cmd_loss=self.get_cmd_loss()
                    invar_loss = cmd_loss
                
                    valid_incon_losses.append(incon_loss.item())
                    valid_invar_losses.append(invar_loss.item())
                    valid_recon_losses.append(recon_loss.item())
                
                cls_loss = self.criterion(y_tilde, labels)
                loss = cls_loss
                
                eval_loss.append(loss.item())

                y_pred.append(y_tilde.detach().cpu().numpy())
                y_true.append(labels.detach().cpu().numpy())

        eval_loss = np.mean(eval_loss)

        y_true = np.concatenate(y_true, axis=0).squeeze()
        y_pred = np.concatenate(y_pred, axis=0).squeeze()
        
        y_pred = np.argmax(y_pred, 1)
        
        return eval_loss, y_true, y_pred
        




    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
