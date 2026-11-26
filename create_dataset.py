from scipy.signal import butter, lfilter

import numpy as np
import pickle
import os
import scipy.io as scio
import random
import math

def to_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj,f)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


class SEED_IV:
    def __init__(self, config):

        session1_label = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3]
        session2_label = [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1]
        session3_label = [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]

        DATA_PATH = str(config.dataset_dir)
        
        SUBJECT = str(config.subject)
        
        if config.exper_setting=='indep':
            train_fn='/indep/train_'+str(SUBJECT)+'.pkl'
            dev_fn='/indep/dev_'+str(SUBJECT)+'.pkl'
            test_fn='/indep/test_'+str(SUBJECT)+'.pkl'
            
            try:
                self.train = load_pickle(DATA_PATH + train_fn)            
                self.dev = load_pickle(DATA_PATH + dev_fn)
                self.test = load_pickle(DATA_PATH + test_fn)    
            except:
                self.train = train = []
                self.dev = dev = []
                self.test = test = []
                sessions=['1','2','3']
                for session in sessions:
                    if session=='1':
                        labels=session1_label
                    elif session=='2':
                        labels=session2_label
                    elif session=='3':
                        labels=session3_label
                    else:
                        print('no such session')
                        exit(0)

                    fns=os.listdir(os.path.join(DATA_PATH,session))
                    for fn in fns:
                        current_subject=fn.split('_')[0]
                        current_exp_path = os.path.join(DATA_PATH,session, fn)
                        data = scio.loadmat(current_exp_path)

                        if current_subject==config.subject:                            
                            for i in range(1,25):
                                key='de_LDS'+str(i)
                                features = data[key]
                                features=features.transpose(1,0,2)
                                for feature in features:
                                    test.append((feature,labels[i-1]))
                        else:
                            for i in range(1,25):
                                key='de_LDS'+str(i)
                                features = data[key]
                                features=features.transpose(1,0,2)
                                for feature in features:
                                    train.append((feature,labels[i-1]))
                     
                random.shuffle(train)
                offset=int(len(train)*0.7)
                self.dev=dev=train[offset:]
                self.train=train=train[:offset]
                
                
                to_pickle(train,DATA_PATH+train_fn)
                to_pickle(dev,DATA_PATH+dev_fn)
                to_pickle(test,DATA_PATH+test_fn)
        else:
            train_fn='/train_'+str(config.subject)+'.pkl'
            dev_fn='/dev_'+str(config.subject)+'.pkl'
            test_fn='/test_'+str(config.subject)+'.pkl'

            try:
                self.train = load_pickle(DATA_PATH + train_fn)
                self.dev = load_pickle(DATA_PATH + dev_fn)
                self.test = load_pickle(DATA_PATH+test_fn)
            except:

                self.train = train = []
                self.dev = dev = []
                self.test = test = []
                sessions=['1','2','3']
                for session in sessions:
                    if session=='1':
                        labels=session1_label
                    elif session=='2':
                        labels=session2_label
                    elif session=='3':
                        labels=session3_label
                    else:
                        print('no such session')
                        exit(0)

                    fns=os.listdir(os.path.join(DATA_PATH,session))
                    for fn in fns:
                        current_subject=fn.split('_')[0]
                        if current_subject==config.subject:
                            current_exp_path = os.path.join(DATA_PATH,session, fn)
                            data = scio.loadmat(current_exp_path)
                            for i in range(1,25):
                                key='de_LDS'+str(i)
                                features = data[key]
                                features=features.transpose(1,0,2)
                                for feature in features:
                                    if i<17:
                                        train.append((feature,labels[i-1]))
                                    else:
                                        test.append((feature,labels[i-1]))
                     
                random.shuffle(train)
                offset=int(len(train)*0.7)
                self.dev=dev=train[offset:]
                self.train=train=train[:offset]
                to_pickle(train,DATA_PATH+train_fn)
                to_pickle(dev,DATA_PATH+dev_fn)
                to_pickle(test,DATA_PATH+test_fn)
                    
    def get_data(self, mode):
        if mode=='train':
            return self.train
        elif mode=='dev':
             return self.dev
        elif mode=='test':
            return self.test
 


class SEED:
    def __init__(self, config):

        DATA_PATH = str(config.dataset_dir)
        SUBJECT = str(config.subject)
        if config.exper_setting=='indep':
            train_fn='/indep/train_'+str(SUBJECT)+'.pkl'
            dev_fn='/indep/dev_'+str(SUBJECT)+'.pkl'
            test_fn='/indep/test_'+str(SUBJECT)+'.pkl'

            try:
                self.train = load_pickle(DATA_PATH + train_fn)            
                self.dev = load_pickle(DATA_PATH + dev_fn)
                self.test = load_pickle(DATA_PATH + test_fn)    
            except:
                self.train = train = []
                self.dev = dev = []
                self.test = test = []
                file_names = os.listdir(DATA_PATH)
                labels = scio.loadmat(os.path.join(DATA_PATH,'label.mat'))['label'][0]+1
                
                for fn in file_names:

                    ext=fn.split('.')[-1]
                    if ext=='pkl':
                        continue
                    if fn in ['readme.txt', 'label.mat', 'indep']:
                        continue
                    exp_subject = fn.split('_')[0]
                    
                    exp_path = os.path.join(DATA_PATH,fn)

                    mat = scio.loadmat(exp_path)
                 
                    for i in range(1, 16):
                        key='de_LDS'+str(i)
                        feature=mat[key]
                        feature = feature.transpose(1,0,2)

                        for fe in feature:
                            if exp_subject==SUBJECT: 
                                test.append((fe,labels[i-1]))
                            else:
                                train.append((fe,labels[i-1]))

                random.shuffle(train)
                offset=int(len(train)*0.7)
                self.dev=dev=train[offset:]
                train=train[:offset]                

                if not os.path.exists(os.path.join(DATA_PATH, 'indep')): 
                    os.makedirs(os.path.join(DATA_PATH, 'indep'))

                to_pickle(train, DATA_PATH+train_fn)
                to_pickle(dev, DATA_PATH+dev_fn)
                to_pickle(test, DATA_PATH+test_fn)

        else:
            train_fn='/train_'+str(SUBJECT)+'.pkl'
            dev_fn='/dev_'+str(SUBJECT)+'.pkl'
            test_fn='/test_'+str(SUBJECT)+'.pkl'
            print("seed test_fn", test_fn)
            
            try:
                self.train = load_pickle(DATA_PATH + train_fn)            
                self.dev = load_pickle(DATA_PATH + dev_fn)
                self.test = load_pickle(DATA_PATH + test_fn)
                
                
            except:
                self.train = train = []
                self.dev = dev = []
                self.test = test = []
                file_names = os.listdir(DATA_PATH)
                labels = scio.loadmat(os.path.join(DATA_PATH,'label.mat'))['label'][0]+1
                for fn in file_names:
                    if fn in ['readme.txt', 'label.mat']:
                        continue
                    exp_subject = fn.split('_')[0]
                    
                    if exp_subject==SUBJECT: 
                        exp_path = os.path.join(DATA_PATH,fn)
                        mat = scio.loadmat(exp_path)
                        for i in range(1, 16):
                            key='de_LDS'+str(i)
                            feature=mat[key]
                            feature = feature.transpose(1,0,2)
                            if i<10:
                                for fe in feature:
                                    train.append((fe,labels[i-1]))
                            else:
                                for fe in feature:
                                    test.append((fe,labels[i-1]))
                random.shuffle(train)
                offset=int(len(train)*0.7)
                self.dev=dev=train[offset:]
                train=train[:offset]
                

                to_pickle(train, DATA_PATH+train_fn)
                to_pickle(dev, DATA_PATH+dev_fn)
                to_pickle(test, DATA_PATH+test_fn)

    def get_data(self, mode):
        if mode=='train':
            return self.train
        elif mode=='dev':
             return self.dev
        elif mode=='test':
            return self.test
 
