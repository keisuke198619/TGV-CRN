import math
import numpy as np
import torch
from torch.utils import data
import cloudpickle

class nbaDataset(data.Dataset):
    def __init__(self, list_IDs, obs_w, args, TEST=False):
        '''Initialization'''
        self.list_IDs = list_IDs
        self.obs_w = obs_w
        self.gamedata_dir = args.gamedata_dir
        self.TEST = TEST
        self.n_games = args.n_games
        self.batchsize = args.batchsize_data
        self.len_seqs = len(list_IDs)
        self.burn_in = args.burn_in
        self.x_dim_permuted = args.x_dim_permuted
        self.n_agents = args.n_agents
        self.vel = args.vel
        self.max_v = args.max_v 

    def __len__(self):
        '''Denotes the total number of samples'''
        return len(self.list_IDs)

    def __getitem__(self, index):
        '''Generates one sample of data'''
        data_dir = self.gamedata_dir
        # Select sample
        ID = self.list_IDs[index]
        if not self.TEST:
            batch_no = ID // self.batchsize
            batch_index = ID % self.batchsize
            with open(data_dir+'_tr_'+str(batch_no)+'.pkl', 'rb') as f:
                data,_,_,_,scores,treatments,lengths = cloudpickle.load(f) # ,allow_pickle=True
            index_ = []
        else:
            J = 8
            batch_no = ID // math.ceil(self.len_seqs/J)
            batch_index = ID % math.ceil(self.len_seqs/J)
            with open(data_dir+'_te_'+str(batch_no)+'.pkl', 'rb') as f:
                data,_,scores,treatments,lengths,indices = cloudpickle.load(f) 
            index_ = indices[batch_index]

        # Load labels
        label = np.array(scores[batch_index])[np.newaxis]
        
        # Load data
        X_all = data[batch_index]
        X_demographic = np.concatenate([np.zeros((1)),X_all[0,-50:]],0) # np.zeros((1))
        posvel = X_all[:,-self.n_agents*4-50:-50].reshape((-1,self.n_agents,4))
        
        if self.vel:
            pos = posvel[:,:,:2]
            vel = posvel[:,:,2:]/self.max_v
            if False: # cos, sin, norm
                vel_norm = np.sqrt(np.sum(vel**2,2))[:,:,np.newaxis]
                nonzeroindex = np.where(vel_norm>1e-3)
                cossin = np.zeros((pos.shape[0],self.n_agents,2))
                cossin[nonzeroindex[0],nonzeroindex[1]] = vel[nonzeroindex[0],nonzeroindex[1]]/np.repeat(vel_norm[nonzeroindex[0],nonzeroindex[1]],2,axis=1)
                poscossinnorm = np.concatenate([pos,cossin,vel_norm],2).reshape((-1,self.n_agents*5))
                # cossinnorm = np.concatenate([cossin,vel_norm],2).reshape((-1,self.n_agents*3))
                X_all = np.concatenate([poscossinnorm,X_all[:,:-self.n_agents*4-50],np.zeros((pos.shape[0],1))],1)
                # X_all = np.concatenate([pos,cossinnorm,X_all[:,:-self.n_agents*4-50],np.zeros((pos.shape[0],1))],1)
            else: # vel_x, vel_y
                posvel = np.concatenate([pos,vel],2).reshape((-1,self.n_agents*4))
                X_all = np.concatenate([posvel,X_all[:,:-self.n_agents*4-51],np.zeros((pos.shape[0],1)),X_all[:,-self.n_agents*4-50,np.newaxis]],1)

        else:
            pos = posvel[:,:,:2].reshape((-1,self.n_agents*2))
            vel = posvel[:,:,2:].reshape((-1,self.n_agents*2))
            
            X_all = np.concatenate([pos,vel,X_all[:,:-self.x_dim_permuted-51],np.zeros((pos.shape[0],1)),X_all[:,-self.x_dim_permuted-50,np.newaxis]],1)
            # X_all = np.concatenate([X_all[:,-self.x_dim_permuted-50:-50],X_all[:,:-self.x_dim_permuted-50]],1)
        
        # X_all:
        # 0-21: players and ball xy, (22-54)vel, 22-32(55-65): other index, 33(66) ballz (not analyzed) 
        # 22-32: shot_area(3),dist_def_feet,max_dist_def_others_feet,shot_prob2,shot_prob_other_max,Clock,ShotClock,Ball_OF_,Ball_Hold_
        
        X_treatment_res = treatments[batch_index].squeeze(1)

        X = torch.from_numpy(X_all.astype(np.float32)) # time,dim
        X_demo = torch.from_numpy(X_demographic.astype(np.float32)) # 0
        X_treatment = torch.from_numpy(X_treatment_res.astype(np.float32)) # time
        y = torch.from_numpy(label.astype(np.float32)).squeeze() # 1

        # lengths
        lengths = torch.from_numpy(lengths[batch_index].astype(np.float32)).squeeze()
        # for counterfactual prediction
        horizon = self.obs_w-self.burn_in
        X_treatment_all = torch.zeros((self.obs_w,horizon-1-5)) 
        for h in range(horizon-1-5):
            X_treatment_all[self.burn_in+h,h] = torch.ones(1)
            
        # no ground truth
        X_all, y_all, X_treatment_opt = [], [], []

        return X, X_demo, X_treatment, y, lengths, index_, X_treatment_all, X.clone(), y_all, X_treatment_opt
