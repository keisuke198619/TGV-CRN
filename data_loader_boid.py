import numpy as np
import torch
from torch.utils import data

class boidDataset(data.Dataset):
    def __init__(self, list_IDs, obs_w, args, train=1, factual_test=None):
        '''Initialization'''
        self.list_IDs = list_IDs
        self.obs_w = obs_w
        self.interv_w = args.intervention_window
        if train==1: data_str = 'train'
        elif train ==0:data_str = 'val'
        else: data_str = 'test'
        if '1' in args.data: 
            verstr = str(1)+'_'
        else: verstr = ''
        self.data_dir = 'simulation/boid/'+data_str+'_boid20_ro1.0_'+verstr+'l21_Fs1/'

        self.burn_in = args.burn_in
        self.x_dim_permuted = args.x_dim_permuted
        self.n_agents = args.n_agents
        self.max_p = args.max_p
        self.train = train
        # self.TEST = TEST
        self.factual_test = factual_test

    def __len__(self):
        '''Denotes the total number of samples'''
        return len(self.list_IDs)

    def __getitem__(self, index):
        '''Generates one sample of data'''
        data_dir = self.data_dir
        # Select sample
        ID = self.list_IDs[index]

        # Load data
        loc, phi, vel, treatment, _ = np.load(data_dir + '{}.npy'.format(ID))

        X_demographic = np.zeros((1))
        X_treatment_res_all = treatment
        N = loc.shape[0]
        for i in range(N):
            if np.sum(treatment[i]==1) > 0:
                treat_ind = np.where(treatment[i]==1)[0][0]
                X_treatment_res_all[i,treat_ind-1] = 1

        # angular velocity (y, outcome)
        angvel = np.zeros((N,self.obs_w))
        center = np.mean(loc,3)
        vec_ic = loc - center[:,:,:,np.newaxis].repeat(self.n_agents,3)
        angvel_all = np.cross(vec_ic,vel,axis=2)
        for i in range(N):
            for t in range(self.obs_w): # self.burn_in+self.interv_w
                angvel[i,t] = np.abs(np.mean(angvel_all[i,t+1:t+2]))
        # angvel_all = angvel_all[:,1:]-angvel_all[:,:-1]
        # vel_polar = np.concatenate([phi[:,:self.obs_w+1],np.sqrt(np.sum(vel[:,:self.obs_w+1,np.newaxis]**2,3))],2) # 
        vel_polar = np.concatenate([np.cos(phi[:,:self.obs_w+1]),np.sin(phi[:,:self.obs_w+1]),np.sqrt(np.sum(vel[:,:self.obs_w+1,np.newaxis]**2,3))],2) # 
        X_all = np.concatenate([loc[:,:self.obs_w+1]/self.max_p,vel_polar,vel[:,:self.obs_w+1]],2).transpose((0,1,3,2)).reshape((N,self.obs_w+1,-1)) 
        X_all = np.concatenate([X_all,np.abs(np.mean(angvel_all[:,:self.obs_w+1,:,np.newaxis],2))],2).transpose((1,2,0))
        label_all = angvel.transpose((1,0))

        X_treatment_res_all = X_treatment_res_all[:,:,0].transpose((1,0))
        if self.train==1: # not self.TEST:
            X = X_all[:,:,0]
            X_treatment_res = X_treatment_res_all[:,0]
            label = label_all[:,0]
            X_treatment_res_opt = np.zeros((1))
        else:
            i = self.factual_test[index]
            X = X_all[:,:,i]
            X_treatment_res = X_treatment_res_all[:,i]
            label = label_all[:,i]
            X_treatment_res_opt = np.array([np.argmax(label_all[-1])])

        X = torch.from_numpy(X.astype(np.float32)) #
        X_demo = torch.from_numpy(X_demographic.astype(np.float32)) # 
        X_treatment = torch.from_numpy(X_treatment_res.astype(np.float32)) # 
        y = torch.from_numpy(label.astype(np.float32)) # 

        # not variable length
        lengths = []

        # for counterfactual prediction
        X_treatment_all = torch.from_numpy(X_treatment_res_all.astype(np.float32)) # 
        # for evaluation of counterfactual prediction
        X_all = torch.from_numpy(X_all.astype(np.float32)) # 
        X_treatment_opt = torch.from_numpy(X_treatment_res_opt.astype(np.float32))
        y_all = torch.from_numpy(label_all.astype(np.float32))

        return X[:-1], X_demo, X_treatment, y, lengths, [], X_treatment_all, X_all[:-1], y_all, X_treatment_opt

