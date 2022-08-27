import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from utils import batch_error, compute_x_ind, compute_collision, sample_gumbel_softmax
from utils import MLP, weights_init
from utils import compute_global

class RNN(nn.Module):
    def __init__(self, n_X_features, n_X_static_features,
                 n_classes, args, hidden_size,
                 num_layers=2, dropout = 0.2):
        super().__init__()

        self.hidden_size = hidden_size
        self.batch_size = args.batch_size
        self.n_X_features = n_X_features
        self.n_X_static_features = n_X_static_features
        self.x_static_emb_size = n_X_static_features if n_X_static_features > 1 else 1
        self.n_classes = n_classes
        self.num_layers = num_layers
        
        self.z_dim = hidden_size
        self.dataset = args.data
        self.obs_w = args.observation_window
        self.x_emb_size = 32
        self.theory = True if "T" in args.model else False
        self.theory2 = True if "TT" in args.model else False
        self.vel = args.vel
        self.x_dim_permuted = args.x_dim_permuted 
        self.x_dim_predicted0 = args.x_dim_predicted
        self.x_dim_predicted = args.x_dim_predicted if self.theory else self.n_X_features

        self.n_agents = args.n_agents
        self.n_dim_each_permuted = int(self.x_dim_permuted//self.n_agents)
        self.n_feat = self.n_dim_each_permuted 
        self.n_dim_each_pred = int(self.x_dim_predicted0//self.n_agents)
        n_feat = self.n_feat
        self.xavier = True
        self.burn_in0 = args.burn_in0
        n_out = 8
        n_all_agents = self.n_agents

        self.n_t_classes = 1
        self.x2emb = nn.Linear(n_X_features, self.x_emb_size)

        self.y_positive = args.y_positive
        self.x_residual = args.x_residual
        self.y_residual = args.y_residual
        self.rollout_y_train = args.rollout_y_train
        if 'boid' in self.dataset:
            self.max_p = args.max_p
        elif 'carla' in self.dataset:
            self.max_p = args.max_p
            self.max_v = args.max_v
            self.max_y = args.max_y
        elif 'nba' in self.dataset:
            self.max_v = args.max_v

        self.dim_rec_global = args.dim_rec_global

        # RNN---------------------
        n_hid = 8
        y_dim = self.n_dim_each_permuted 
        self.z_dim_each = 4 
        n_layers = 2

        self.n_X_unpermute = self.n_X_features - self.n_dim_each_permuted*n_all_agents
        
        h_dim = 64

        # RNN 
        self.rnn_f = nn.GRUCell(input_size=self.x_emb_size + 1 + self.z_dim, hidden_size=hidden_size)

        # DSW-------------------------------------
        self.concat_f = nn.Linear(hidden_size * 2, hidden_size)
        self.x_static2emb = nn.Linear(n_X_static_features, self.x_static_emb_size)
        self.x2emb = nn.Linear(n_X_features, self.x_emb_size)

        # Outcome
        self.hidden2hidden_outcome_f = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.x_static_emb_size + 1 + h_dim, hidden_size),
            nn.Dropout(0.3),
            nn.ReLU(),
        )
        if self.y_positive:
            self.hidden2out_outcome_f = nn.Sequential(
                nn.Linear(hidden_size, self.n_classes, bias=False),
                nn.Softplus()
            )
        else:
            self.hidden2out_outcome_f = nn.Linear(hidden_size, self.n_classes, bias=False)
       
        # covariates
        self.hidden2hidden_x = nn.Sequential( # nn.Dropout(0.5), nn.Dropout(0.3),
            nn.Linear(self.x_static_emb_size + 1 + h_dim, hidden_size),
            nn.Softplus(),
        )
        self.hidden2out_x = nn.Linear(hidden_size, self.x_dim_predicted, bias=False)

        if 'carla' in self.dataset:
            self.collision = nn.Sequential(
                nn.Linear(4, 8), # 1+
                nn.Softplus(),
                nn.Linear(8, 1), # 
                nn.Sigmoid(),
            )

        self.self_1 = [self.dataset, self.n_dim_each_permuted, self.vel, self.n_agents, self.theory, self.x_dim_predicted, self.x_dim_permuted]
        if self.dataset == 'nba':
            self.self_2 = [self.theory2, self.max_v]
        elif 'carla' in self.dataset:
            self.self_2 = [self.max_v, self.max_p, self.max_y]
        elif 'boid' in self.dataset:
            self.self_2 = self.max_p
    
    def compute_collisions(self,input,batchSize,device,x_dim,n_agents):
        temperature = 0.1
        tmp_collision = torch.zeros(batchSize,self.n_agents-1,2).to(device)
        for k in range(1,n_agents):
            input_collision = compute_collision(input,x_dim,n_agents,k)
            tmp_collision[:,k-1,1:2] = torch.sigmoid((input_collision[:,0:1] - 0.5/self.max_p - 2/self.max_p*self.collision(input_collision[:,1:]))*self.max_p)
            tmp_collision[:,k-1,0] = 1-tmp_collision[:,k-1,1]
        return torch.max(tmp_collision[:,:,0],1)[0]

    def forward(self, x__, x_demo, f_treatment, y_, cf_treatment=None, burn_in=None, Train=False, lengths=None): # , target=None):
        torch.autograd.set_detect_anomaly(True)
        # added
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        batchSize = x__.size(0)
        n_all_agents = self.n_agents
        n_agents = self.n_agents
        n_feat = self.n_feat
        x_dim = self.n_dim_each_permuted
        len_time = x__.size(1)-1
        

        if self.dataset == 'nba' or self.dataset == 'carla':
            x_demo_emd = self.x_static2emb(x_demo[:,0:1])
        else:
            x_demo_emd = self.x_static2emb(x_demo)
        f_hx = torch.randn(x__.size(0), self.hidden_size)
        f_old = f_hx
        f_outputs = []
        f_zxs = []
        f_outcome_out = []
        f_h = []
        CF_flag = (cf_treatment is not None)
        if CF_flag: 
            if self.dataset == 'nba':
                horizon = self.obs_w - len_time 
            else:
                horizon = 0 
            n_cf = cf_treatment.shape[2]
            cf_hx = torch.randn(n_cf, x__.size(0), self.hidden_size)
            cf_old = cf_hx
            cf_outputs = [[] for _ in range(n_cf)]
            cf_outcome_out = [[] for _ in range(n_cf)]
            x_cf = [torch.cat([x__.clone(),torch.zeros(batchSize,horizon,self.n_X_features)],1) for n in range(n_cf)]
            y_cf = [torch.cat([y_.clone(),torch.zeros(batchSize,horizon)],1) for n in range(n_cf)]
            x = torch.cat([x__.clone(),torch.zeros(batchSize,horizon,self.n_X_features)],1)
            y = torch.cat([y_.clone(),torch.zeros(batchSize,horizon)],1)
            x_ = x.clone()
        else:
            cf_outcome_out = None
            x_ = x__.clone()
            x = x__.clone()
            y = y_.clone()
            horizon = 0 

        if burn_in is None:
            rollout = False
            burn_in = len_time
        else:
            rollout = True

        if self.theory: 
            ind_x, ind_x_, ind_x0, ind_x1 = compute_x_ind(self.x_dim_predicted,self.dataset,n_agents, vel=self.vel)
        else:
            ind_x, ind_x_, ind_x0, ind_x1 = compute_x_ind(self.x_dim_predicted0,self.dataset,n_agents, vel=self.vel)

        for i in range(len_time+horizon): # time
            x_0 = x[:, i].clone()
            x_1 = x[:, i+1].clone()

            x_emb = self.x2emb(x[:, i, :])
            f_zx = torch.cat((x_emb, f_old), -1)

            f_zxs.append(f_zx) # q in the paper
            if i < len_time:
                f_inputs = torch.cat((f_zx, f_treatment[:,i].unsqueeze(1)), -1) # [128,225]
                f_hx = self.rnn_f(f_inputs, f_hx)
                f_outputs.append(f_hx)

                # Y
                f_hidden = torch.cat((f_zx, x_demo_emd, f_treatment[:,i:i+1]), -1)
                f_h_ = self.hidden2hidden_outcome_f(f_hidden) 
                f_h.append(f_h_)
                
                # X
                f_x_ = self.hidden2hidden_x(f_hidden) 

                if self.x_residual: 
                    f_x_ = self.hidden2out_x(f_x_)
                    x_new = torch.zeros(batchSize,self.x_dim_predicted).to(device)
                    x_new[:,ind_x] = f_x_[:,ind_x] + x_0[:, ind_x0]
                    x_new[:,ind_x_] = f_x_[:,ind_x_] 
                else:
                    x_new = self.hidden2out_x(f_x_) 

                if i >= self.burn_in0:
                    # compute global parameters
                    tmp_pred = compute_global(x_new,x[:,i],x[:,i-1],x[:,i+1],f_treatment[:,i:i+1],x_demo,device,i,self.self_1,self.self_2,CF_flag=False)
                    if self.theory and 'carla' in self.dataset: # collision
                        input_collision = x[:,i] # tmp_pred
                        collision = self.compute_collisions(input_collision,batchSize,device,x_dim,n_agents) 
                        tmp_pred[:,-1] = collision

                    if torch.abs(tmp_pred[0,0])>200:
                        import pdb; pdb.set_trace()

                    # outcome 
                    if i == 0 or not self.y_residual:
                        y_new = self.hidden2out_outcome_f(f_h_) 
                    else:
                        if self.theory and 'carla' in self.dataset: 
                            y_new = 0.01*(1-collision.unsqueeze(1))*self.hidden2out_outcome_f(f_h_) + x[:,i:i+1,-4] #y[:,i:i+1] # 0->1, 1->0
                        else:
                            y_new = self.hidden2out_outcome_f(f_h_) + y[:,i:i+1]
                    if torch.sum(torch.isnan(y_new))>0:
                        import pdb; pdb.set_trace()

                    f_outcome_out.append(y_new)
                else:
                    f_outcome_out.append(y[:,i+1:i+2])

            if CF_flag: 
                for n in range(n_cf):
                    x_0_cf = x_cf[n][:, i].clone()
                    x_1_cf = x_cf[n][:, i+1].clone()

                    cf_x_emb = self.x2emb(x_cf[n][:, i, :])
                    cf_zx = torch.cat((cf_x_emb, cf_old[n]), -1)

                    cf_inputs = torch.cat((cf_zx, cf_treatment[:,i:i+1,n]), -1)

                    # RNN
                    cf_hx[n] = self.rnn_f(cf_inputs, cf_hx[n])
                    cf_outputs[n].append(cf_hx[n])

                    # added
                    cf_hidden = torch.cat((cf_zx, x_demo_emd, cf_treatment[:,i:i+1,n]), -1)
                    cf_h = self.hidden2hidden_outcome_f(cf_hidden)
                    cf_x_ = self.hidden2hidden_x(f_hidden) 

                    if self.x_residual: 
                        cf_x_ = self.hidden2out_x(cf_x_)
                        x_new_cf = torch.zeros(batchSize,self.x_dim_predicted).to(device)
                        x_new_cf[:,ind_x] = cf_x_[:,ind_x] + x_0_cf[:, ind_x0]
                        x_new_cf[:,ind_x_] = cf_x_[:,ind_x_] 
                    else:
                        x_new_cf = self.hidden2out_x(cf_x_) 

                    if i >= self.burn_in0:
                        # compute global parameters
                        tmp_pred_cf = compute_global(x_new_cf,x_cf[n][:,i],x_cf[n][:,i-1],x_cf[n][:,i+1],cf_treatment[:,i:i+1,n],x_demo,device,i,self.self_1,self.self_2,CF_flag=True)
                        if self.theory and 'carla' in self.dataset: # collision
                            input_collision = x_cf[n][:,i] # tmp_pred_cf
                            collision = self.compute_collisions(input_collision,batchSize,device,x_dim,n_agents) 
                            tmp_pred_cf[:,-1] = collision
                    
                        # outcome
                        if i == 0 or not self.y_residual:
                            y_new_cf = self.hidden2out_outcome_f(cf_h) 
                            if 'nba' in self.dataset: 
                                y_new_cf = torch.clamp(y_new_cf,max=1)
                        else:
                            if self.theory and 'carla' in self.dataset and i >= self.burn_in0: 
                                y_new_cf = 0.01*(1-collision.unsqueeze(1))*self.hidden2out_outcome_f(cf_h) + x_cf[n][:,i:i+1,-4] #y_cf[n][:,i:i+1] # 0->1, 1->0
                            else:
                                y_new_cf = self.hidden2out_outcome_f(cf_h) + y_cf[n][:,i:i+1]
                        cf_outcome_out[n].append(y_new_cf)
                        # roll out
                        if rollout and i >= burn_in -1: # 
                            x_cf[n][:,i+1] = tmp_pred_cf
                            y_cf[n][:,i+1] = y_new_cf.squeeze(1)
                        elif not rollout and i >= self.burn_in0:
                            x_cf_[n][:,i+1] = tmp_pred_cf
                    else:
                        cf_outcome_out[n].append(y_cf[n][:,i+1:i])
            # roll out
            if rollout and i >= burn_in-1 and i < len_time: # 
                x[:,i+1] = tmp_pred
                y[:,i+1] = y_new.squeeze(1)
            elif not rollout and i >= self.burn_in0:
                x_[:,i+1] = tmp_pred
                if self.rollout_y_train and i >= burn_in -1:
                    y[:,i+1] = y_new.squeeze(1)
                    x[:,i+1] = tmp_pred

        if CF_flag: 
            cf_outcome_out = torch.stack(y_cf, dim=0)[:,:,1:].permute(0,2,1).unsqueeze(3)

        if CF_flag:
            x_out_cf = torch.stack(x_cf, dim=0) # if rollout else torch.stack(x_cf_, dim=0)
            x_out = x if rollout else x_
        else:
            x_out = x if rollout else x_
            x_out_cf = torch.zeros(1,1).to(device)

        # dummy 
        L_kl = torch.zeros(1).to(device)
        a_or_ipw_outputs = [torch.zeros(1).to(device)]
        
        return a_or_ipw_outputs, f_outcome_out, cf_outcome_out, f_h, L_kl, x_out, x_out_cf