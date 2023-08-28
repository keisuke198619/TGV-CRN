import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from utils import GradientReversal
from utils import cudafy_list, sample_gauss, nll_gauss, kld_gauss, batch_error, sample_gumbel_softmax
from utils import Attn, MLP, weights_init, edge2node, node2edge, encode_onehot, compute_x_ind, compute_collision 
from utils import compute_global

class GVCRN(nn.Module):
    def __init__(self, n_X_features, n_X_static_features,
                 n_classes, args, hidden_size,
                 num_layers=2, dropout = 0.2):
        super().__init__()

        self.hidden_size = hidden_size
        self.batch_size = args.batch_size
        self.n_X_features = n_X_features
        self.n_X_static_features = n_X_static_features
        self.n_classes = n_classes
        self.num_layers = num_layers
        self.x_static_emb_size = n_X_static_features if n_X_static_features > 1 else 1
        # self.z_dim = n_Z_confounders
        self.dataset = args.data
        self.obs_w = args.observation_window
        self.variable_length = args.variable_length
        self.vel = args.vel

        self.GraphNN = True if "G" in args.model else False
        self.balancing = True if "B" in args.model else False
        self.theory = True if "T" in args.model else False
        self.theory2 = True if "TT" in args.model else False
        self.negativeGradient = True if "Ne" in args.model else False
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

        # VRNN---------------------
        n_hid = 8
        y_dim = self.n_dim_each_permuted 
        self.z_dim_each = 4 
        n_layers = 2

        # if self.VariationalRNN:
        self.n_gnn = 3 # 1: RNN, 3: VRNN
        rnn_micro_dim = 64
        self.n_X_unpermute = self.n_X_features - self.n_dim_each_permuted*n_all_agents

        h_dim0 = 64
        if self.GraphNN:
            h_dim = self.n_X_unpermute # self.x_emb_size # 
            h_dim4 = 0
        else:
            h_dim = h_dim0
            h_dim4 = h_dim
        
        n_agents = n_all_agents
        n_out = n_hid
        self.n_layers = n_layers
        self.rnn_micro_dim = rnn_micro_dim       

        in_prior = self.n_X_unpermute + self.n_classes if self.GraphNN else n_X_features + self.n_classes
        in_enc = in_prior
        z_dim = self.z_dim_each*n_all_agents + self.n_X_unpermute if self.GraphNN else n_X_features
        # in_dec = z_dim
        in_dec = self.n_X_unpermute if self.GraphNN else n_X_features

        self.bn_enc = nn.BatchNorm1d(h_dim)
        self.bn_prior = nn.BatchNorm1d(h_dim)
        
        if self.GraphNN:
            out_prior = n_out*n_all_agents + rnn_micro_dim*(n_agents) #  + h_dim # +1
            h_dim2 = 0
            h_dim3 = 0
            if not self.theory:
                out_prior += h_dim0 + self.n_X_unpermute
                h_dim2 += h_dim0
                h_dim3 += h_dim0

                self.prior_d = nn.Sequential(
                    nn.Linear(in_prior, h_dim),
                    nn.ReLU(),
                    nn.Linear(h_dim, h_dim),
                    nn.ReLU())
                self.enc_d = nn.Sequential(
                    nn.Linear(in_enc, h_dim),
                    nn.ReLU(),
                    nn.Linear(h_dim, h_dim),
                    nn.ReLU())
                self.dec = nn.Sequential(
                    nn.Linear(in_dec, h_dim0),
                    nn.ReLU(),
                    nn.Linear(h_dim0, h_dim0),
                    nn.ReLU())
                self.bn_dec = nn.BatchNorm1d(h_dim0)

            out_dec = n_out*n_all_agents 
            
        else:
            out_prior = rnn_micro_dim*(n_agents) + h_dim # +1
            out_dec = 0
            h_dim2 = h_dim
            h_dim3 = 0
            if not self.theory:
                out_prior += h_dim0

            self.prior = nn.Sequential(
                nn.Linear(in_prior, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, h_dim),
                nn.ReLU())
            self.enc = nn.Sequential(
                nn.Linear(in_enc, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, h_dim),
                nn.ReLU())
            self.dec = nn.Sequential(
                nn.Linear(in_dec, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, h_dim),
                nn.ReLU())
            self.bn_dec = nn.BatchNorm1d(h_dim)

        out_enc = out_prior 
        
        self.enc_mean = nn.Linear(out_enc, z_dim) 
        self.enc_std = nn.Sequential(
            nn.Linear(out_enc, z_dim),
            nn.Softplus()) 

        self.prior_mean = nn.Linear(out_prior, z_dim)
        self.prior_std = nn.Sequential(
            nn.Linear(out_prior, z_dim),
            nn.Softplus())
        
        self.dec_mean = nn.Sequential( 
            nn.Linear(out_dec+self.n_classes+h_dim2+ self.x_static_emb_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, self.x_dim_predicted, bias=False)
        )
        self.dec_mean.apply(weights_init)

        if self.GraphNN:
            self.gru_micro = nn.GRU(y_dim+self.z_dim_each, rnn_micro_dim, n_layers)
            if not self.theory:
                self.gru_micro2 = nn.GRU(self.n_X_unpermute*2, rnn_micro_dim, n_layers)
        else:
            self.gru_micro = nn.GRU(y_dim+z_dim, rnn_micro_dim, n_layers)
            if not self.theory:
                self.gru_micro2 = nn.GRU(self.n_X_unpermute+z_dim, rnn_micro_dim, n_layers)
        
        if self.GraphNN:
            self.x_emb_graph_size = out_enc #
        else:
            self.x_emb_graph_size = 0

        # balancing -------------------------------------
        self.x_static2emb = nn.Linear(n_X_static_features, self.x_static_emb_size)

        # self.balancing or IPW
        if self.balancing and not self.negativeGradient:
            self.hidden2out_a_or_ipw = nn.Sequential(GradientReversal(),
                nn.Linear(out_dec + h_dim4 + self.x_static_emb_size + h_dim3, self.n_t_classes, bias=False), # out_prior
                nn.ReLU(),
            )
        else:
            self.hidden2out_a_or_ipw = nn.Sequential(
                nn.Linear(out_dec + h_dim4 + self.x_static_emb_size + h_dim3, self.n_t_classes, bias=False), # out_prior
                nn.ReLU(),
            )
        
        # Outcome
        y_additional_dim = 1 if self.GraphNN else 1+self.n_X_unpermute # self.dim_rec_global # 2#   
        if self.y_positive:
            self.hidden2out_outcome_f = nn.Sequential(
                nn.Linear(self.x_static_emb_size + out_dec + h_dim + h_dim3 + y_additional_dim, self.n_classes, bias=False),
                nn.Softplus()
            )
        else:
            self.hidden2out_outcome_f = nn.Linear(self.x_static_emb_size + out_dec + h_dim + h_dim3 + y_additional_dim, self.n_classes, bias=False)
        
        # GNN--------------------------------------------
        if self.GraphNN:
            # self.factor = True
            if self.dataset == 'nba' or self.dataset == 'carla':
                self.n_edge_type = 2
                self.n_node_type = 2
            else:
                self.n_edge_type = 0
                self.n_node_type = 0            
            
            self.mlp1 = nn.ModuleList([nn.ModuleList() for i in range(self.n_gnn)])  
            self.mlp2 = nn.ModuleList([nn.ModuleList() for i in range(self.n_gnn)])  
            self.mlp3 = nn.ModuleList([nn.ModuleList() for i in range(self.n_gnn)])  

            for ped in range(self.n_gnn): # 0: prior, 1:encoder, 2:decoder
                if ped <= 1:
                    n_in = n_feat # +rnn_micro_dim #*n_all_agents # 
                elif ped == 2:
                    n_in = self.z_dim_each # +rnn_micro_dim 

                do_prob = 0

                self.mlp1[ped] = MLP(n_in, n_hid, n_hid, do_prob)
                self.mlp2[ped] = MLP(n_hid * 2 + self.n_edge_type, n_hid, n_hid, do_prob)
                self.mlp3[ped] = MLP(n_hid + self.n_node_type, n_hid, n_out, do_prob)

                self.mlp1[ped].apply(weights_init)
                self.mlp2[ped].apply(weights_init)
                self.mlp3[ped].apply(weights_init)

            # rel_rec, rel_send with numpy 
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            off_diag = np.ones([n_all_agents, n_all_agents]) - np.eye(n_all_agents)

            rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
            rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
            rel_rec = torch.FloatTensor(rel_rec).to(device)
            rel_send = torch.FloatTensor(rel_send).to(device)

            if self.dataset == 'nba':
                # corresponding with rel_rec/rel_send 
                edge_type = torch.zeros(1,n_all_agents*(n_all_agents-1),2).to(device)
                nonball_index = torch.arange(0,n_all_agents*(n_all_agents-1),(n_all_agents-1))[1:-1]
                edge_type[:,nonball_index,0] = 1 # non-ball -> non-ball: [1 0] (receiver) :-(n_all_agents-1)?
                edge_type[:,-(n_all_agents-1):,1] = 1 # non-ball -> ball: [0 1] (sender) OK

                node_type = torch.zeros(1,n_all_agents,2).to(device)
                node_type[:,:-1,0] = 1 # non-ball: [1 0] OK
                node_type[:,-1,1] = 1 # ball: [0 1] OK
            elif self.dataset == 'carla':
                edge_type = torch.zeros(1,n_all_agents*(n_all_agents-1),2).to(device)
                edge_type[:,:(n_all_agents-1),0] = 1 # ego -> ego: [1 0] (receiver) OK
                ego_index = torch.where(rel_send[:,0]==1)[0]
                edge_type[:,ego_index,1] = 1 # ego -> non-ego: [0 1] (sender) (n_all_agents-1):?

                node_type = torch.zeros(1,n_all_agents,2).to(device)
                node_type[:,0,0] = 1 # ego: [1 0] OK 
                node_type[:,1:,1] = 1 # non-ego: [0 1] OK
            else:
                edge_type = None
                node_type = None
            self.rel_rec = rel_rec
            self.rel_send = rel_send
            self.edge_type = edge_type
            self.node_type = node_type

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

    def GNN(self, x, ped): # 0: prior, 1:encoder, 2:decoder
        rel_rec, rel_send = self.rel_rec, self.rel_send
        edge_type, node_type = self.edge_type, self.node_type
        batchSize = x.shape[0]
        if self.edge_type is not None:
            edge_type = edge_type.repeat(batchSize,1,1)
            node_type = node_type.repeat(batchSize,1,1)

        x = self.mlp1[ped](x)  # 2-layer ELU net per node # [128, 5, 8]

        x = node2edge(x, rel_rec, rel_send, edge_type) # [batch, K(K-1), 16]
        x = self.mlp2[ped](x)

        x = edge2node(x, rel_rec, rel_send, node_type)
        x = self.mlp3[ped](x)

        return x 
    # ===================================================================

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

        h_micro = [torch.zeros(self.n_layers, batchSize, self.rnn_micro_dim).to(device) for i in range(n_agents)]
        if not self.theory:
            h_micro2 = torch.zeros(self.n_layers, batchSize, self.rnn_micro_dim).to(device)
        L_kl = torch.zeros(1).to(device)
        if CF_flag:
            h_micro_cf = [torch.zeros(n_cf,self.n_layers, batchSize, self.rnn_micro_dim).to(device) for i in range(n_agents)]
            if not self.theory:
                h_micro2_cf = torch.zeros(n_cf,self.n_layers, batchSize, self.rnn_micro_dim).to(device)
   
        if not self.variable_length:
            non_nan = None
            mean_time_length = len_time-self.burn_in0-1
        else:
            mean_time_length = torch.mean(lengths-1)-self.burn_in0-1

        a_or_ipw_outputs = []
        if self.theory: 
            ind_x, ind_x_, ind_x0, ind_x1 = compute_x_ind(self.x_dim_predicted,self.dataset,n_agents, vel=self.vel)
        else:
            ind_x, ind_x_, ind_x0, ind_x1 = compute_x_ind(self.x_dim_predicted0,self.dataset,n_agents, vel=self.vel)

        
        n_agents_ = n_agents if self.theory else n_agents+1 # or self.GraphNN 

        for i in range(len_time+horizon): # time
            x_0 = x[:, i].clone()
            x_1 = x[:, i+1].clone()
            if self.variable_length:
                non_nan = lengths>=self.obs_w-i

            # VRNN------------------------------------
            micro_in = [[] for _ in range(n_agents_)]# +1

            for ii in range(n_agents_):#+1
                micro_in[ii] = h_micro[ii][-1] if ii < n_agents else h_micro2[-1]
            micro_in = torch.stack(micro_in,dim=1)

            # prior & encorder
            if self.GraphNN:
                x_0_ = x_0[:, :self.x_dim_permuted].reshape(batchSize,n_all_agents,n_feat)
                x_1_ = x_1[:, :self.x_dim_permuted].reshape(batchSize,n_all_agents,n_feat)                
                prior_in = x_0_
                enc_in = x_1_ 
                prior_t = self.GNN(prior_in, ped=0)
                enc_t = self.GNN(enc_in, ped=1)
                if not self.theory:
                    prior_in_ = torch.cat([x_0[:, self.x_dim_permuted:],y[:,i:i+1]],1)
                    enc_in_ = torch.cat([x_1[:, self.x_dim_permuted:],y[:,i+1:i+2]],1)
                    prior_t_ = self.prior_d(prior_in_)
                    prior_t_ = self.bn_prior(prior_t_)
                    enc_t_ = self.enc_d(enc_in_)
                    enc_t_ = self.bn_enc(enc_t_)
            else:
                prior_in_ = torch.cat([x_0,y[:,i:i+1]],1)
                enc_in_ = torch.cat([x_1,y[:,i+1:i+2]],1)
                prior_t_ = self.prior(prior_in_)
                prior_t_ = self.bn_prior(prior_t_)
                enc_t_ = self.enc(enc_in_)
                enc_t_ = self.bn_enc(enc_t_)

            if self.GraphNN:
                if self.theory:
                    prior_t = torch.cat([prior_t.reshape(batchSize,-1),micro_in.reshape(batchSize,-1)],1) # prior_t_
                    enc_t = torch.cat([enc_t.reshape(batchSize,-1),micro_in.reshape(batchSize,-1)],1) # enc_t_
                else:
                    prior_t = torch.cat([prior_t.reshape(batchSize,-1),prior_t_,micro_in.reshape(batchSize,-1)],1) # 
                    enc_t = torch.cat([enc_t.reshape(batchSize,-1),enc_t_,micro_in.reshape(batchSize,-1)],1) # 
                
            else:
                prior_t = torch.cat([prior_t_,micro_in.reshape(batchSize,-1)],1)
                enc_t = torch.cat([enc_t_,micro_in.reshape(batchSize,-1)],1)
            
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t)

            # sample
            if not Train:
                z_t = sample_gauss(enc_mean_t, enc_std_t) # different between forward and sample
            else:
                z_t = sample_gauss(prior_mean_t, prior_std_t)

            # objective function
            if i >= self.burn_in0:
                L_kl += kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t,index=non_nan)
                if torch.isnan(L_kl):
                    import pdb; pdb.set_trace()

            del micro_in

            # decoder 
            if self.GraphNN:
                z_t_ = z_t[:,:n_all_agents*self.z_dim_each].reshape(batchSize,n_all_agents,self.z_dim_each)
                dec_in = z_t_ # torch.cat([, micro_in],dim=2)
                dec_t = self.GNN(dec_in, ped=2).reshape(batchSize,-1)
                if not self.theory:
                    dec_in_ = z_t[:,n_all_agents*self.z_dim_each:]
                    dec_t_ = self.dec(dec_in_) # 
                    dec_t_ = self.bn_dec(dec_t_) # 
            else:
                dec_in_ = z_t
                dec_t = self.dec(dec_in_) # dec_t_
                dec_t = self.bn_dec(dec_t) # dec_t_

            if i < len_time:
                if self.theory or (not self.GraphNN and not self.theory):
                    dec_t_cat = torch.cat([dec_t, x_demo_emd,f_treatment[:,i].unsqueeze(1)],-1)
                else:
                    dec_t_cat = torch.cat([dec_t,dec_t_,x_demo_emd,f_treatment[:,i].unsqueeze(1)],-1)
                if self.x_residual: 
                    dec_mean_t_ = self.dec_mean(dec_t_cat)
                    dec_mean_t = torch.zeros(batchSize,self.x_dim_predicted).to(device)
                    dec_mean_t[:,ind_x] = dec_mean_t_[:,ind_x] + x_0[:, ind_x0]
                    dec_mean_t[:,ind_x_] = dec_mean_t_[:,ind_x_] 
                else:
                    dec_mean_t = self.dec_mean(dec_t_cat)

                # compute global parameters
                if i >= self.burn_in0:
                    tmp_pred = compute_global(dec_mean_t,x[:,i],x[:,i-1],x[:,i+1],f_treatment[:,i:i+1],x_demo,device,i,self.self_1,self.self_2,CF_flag=False)
                    if self.theory:
                        if 'carla' in self.dataset: # collision
                            input_collision = x[:,i] # tmp_pred
                            collision = self.compute_collisions(input_collision,batchSize,device,x_dim,n_agents) 
                            tmp_pred[:,-1] = collision
                            # print('mean factual collision: '+str(torch.mean(collision)))

                    # Y
                    f_hidden = torch.cat([dec_t_cat,x[:,i,self.x_dim_permuted:]],-1) # x_emb 

                    f_h_ = f_hidden

                    if i == 0 or not self.y_residual:
                        y_new = self.hidden2out_outcome_f(f_h_)
                        if 'nba' in self.dataset: 
                            y_new = torch.clamp(y_new,max=1)
                    else:
                        if 'carla' in self.dataset and self.theory:
                            # y_new = 0.01*(1-collision.unsqueeze(1))*self.hidden2out_outcome_f(f_h_) + x[:,i:i+1,-4] # y[:,i:i+1] # 0->1, 1->0
                            y_new = 0.01*(1-collision.unsqueeze(1))*(self.hidden2out_outcome_f(f_h_)+ y[:,i:i+1]) + x[:,i:i+1,-4]
                        else:
                            y_new = self.hidden2out_outcome_f(f_h_) + y[:,i:i+1]
                    if torch.sum(torch.isnan(y_new))>0:
                        import pdb; pdb.set_trace()
                    f_outcome_out.append(y_new)
                    f_h.append(f_h_)

                    # a_or_ipw
                    if self.GraphNN and not self.theory:
                        f_hidden = torch.cat([dec_t,dec_t_, x_demo_emd],-1)#dec_t_,x_emb
                    else:
                        f_hidden = torch.cat([dec_t, x_demo_emd],-1)
                    h = f_hidden

                    out = self.hidden2out_a_or_ipw(h)
                    a_or_ipw_outputs.append(out)
                else:
                    f_outcome_out.append(y[:,i+1:i+2])
                    a_or_ipw_outputs.append(torch.zeros(batchSize,1).to(device))

            if CF_flag: 
                for n in range(n_cf):
                    x_0_cf = x_cf[n][:, i].clone()
                    x_1_cf = x_cf[n][:, i+1].clone()
                    if self.GraphNN:
                        x_0_cf_ = x_0_cf[:, :self.x_dim_permuted].reshape(batchSize,n_all_agents,n_feat)
                        x_1_cf_ = x_1_cf[:, :self.x_dim_permuted].reshape(batchSize,n_all_agents,n_feat)                
                    
                    # VRNN------------------------------------
                    micro_in_cf = [[] for _ in range(n_agents_)]#+1

                    for ii in range(n_agents_):#+1
                        micro_in_cf[ii] = h_micro_cf[ii][n,-1] if ii < n_agents else h_micro2_cf[n,-1]
                    micro_in_cf = torch.stack(micro_in_cf,dim=1)

                    # prior 
                    if self.GraphNN:
                        prior_in_cf = x_0_cf_
                        prior_t_cf = self.GNN(prior_in_cf, ped=0)
                        # prior_t_cf = self.bn_prior(prior_t_cf)
                        
                        if not self.theory:
                            prior_in_cf_ = torch.cat([x_0_cf[:, self.x_dim_permuted:],y_cf[n][:,i:i+1]],1)
                            prior_t_cf_ = self.prior_d(prior_in_cf_)
                            prior_t_cf_ = self.bn_prior(prior_t_cf_)
                    else:
                        prior_in_cf_ = torch.cat([x_0_cf,y_cf[n][:,i:i+1]],1)
                        prior_t_cf = self.prior(prior_in_cf_)
                        prior_t_cf = self.bn_prior(prior_t_cf)

                    if self.GraphNN:
                        if self.theory or (not self.GraphNN and not self.theory):
                            prior_t_cf = torch.cat([prior_t_cf.reshape(batchSize,-1),micro_in_cf.reshape(batchSize,-1)],1) # prior_t_cf_,
                        else:
                            prior_t_cf = torch.cat([prior_t_cf.reshape(batchSize,-1),prior_t_cf_,micro_in_cf.reshape(batchSize,-1)],1) # 
                    else:
                        prior_t_cf = torch.cat([prior_t_cf,micro_in_cf.reshape(batchSize,-1)],1)
                    prior_mean_t_cf = self.prior_mean(prior_t_cf)
                    prior_std_t_cf = self.prior_std(prior_t_cf)

                    # sample
                    z_t_cf = sample_gauss(prior_mean_t_cf, prior_std_t_cf)

                    del micro_in_cf
                    del prior_mean_t_cf, prior_std_t_cf

                    # decoder 
                    if self.GraphNN:
                        z_t_cf_ = z_t_cf[:,:n_all_agents*self.z_dim_each].reshape(batchSize,n_all_agents,self.z_dim_each)
                        dec_in_cf = z_t_cf_ # torch.cat([, micro_in],dim=2)
                        dec_t_cf = self.GNN(dec_in_cf, ped=2).reshape(batchSize,-1)
                        if not self.theory:
                            dec_in_cf_ = z_t_cf[:,n_all_agents*self.z_dim_each:]
                            dec_t_cf_ = self.dec(dec_in_cf_)
                            dec_t_cf_ = self.bn_dec(dec_t_cf_)
                    else:
                        dec_in_cf_ = z_t_cf
                        dec_t_cf = self.dec(dec_in_cf_)
                        dec_t_cf = self.bn_dec(dec_t_cf)

                    if self.theory or (not self.GraphNN and not self.theory):
                        dec_t_cf = torch.cat([dec_t_cf, x_demo_emd,cf_treatment[:,i:i+1,n]],-1)
                    else: 
                        dec_t_cf = torch.cat([dec_t_cf,dec_t_cf_,x_demo_emd,cf_treatment[:,i:i+1,n]],-1) # 

                    if self.x_residual: 
                        cf_x_ = self.dec_mean(dec_t_cf)
                        if torch.sum(torch.abs(cf_x_))<0.1:
                            import pdb; pdb.set_trace()
                        dec_mean_t_cf = torch.zeros(batchSize,self.x_dim_predicted).to(device)
                        dec_mean_t_cf[:,ind_x] = cf_x_[:,ind_x] + x_0_cf[:, ind_x0]
                        dec_mean_t_cf[:,ind_x_] = cf_x_[:,ind_x_] 
                    else:
                        dec_mean_t_cf = self.dec_mean(dec_t_cf)
                    
                    if i >= self.burn_in0:
                        # compute global parameters
                        tmp_pred_cf = compute_global(dec_mean_t_cf,x_cf[n][:,i],x_cf[n][:,i-1],x_cf[n][:,i+1],cf_treatment[:,i:i+1,n],x_demo,device,i,self.self_1,self.self_2,CF_flag=True)
                        if self.theory and 'carla' in self.dataset: # collision
                            input_collision = x_cf[n][:,i] # tmp_pred_cf
                            collision = self.compute_collisions(input_collision,batchSize,device,x_dim,n_agents) 
                            tmp_pred_cf[:,-1] = collision
                            #print('mean counterfactual collision: '+str(torch.mean(collision)))
                    
                        # Y
                        cf_hidden = torch.cat([dec_t_cf,x_cf[n][:,i,self.x_dim_permuted:]],-1)
                        cf_h = cf_hidden

                        if i == 0 or not self.y_residual:
                            y_new_cf = self.hidden2out_outcome_f(cf_h) 
                            if 'nba' in self.dataset: 
                                y_new_cf = torch.clamp(y_new_cf,max=1)
                        else:
                            if self.theory and 'carla' in self.dataset: 
                                # y_new_cf = 0.01*(1-collision.unsqueeze(1))*self.hidden2out_outcome_f(cf_h) + x_cf[n][:,i:i+1,-4] #  y_cf[n][:,i:i+1] # 0->1, 1->0
                                y_new_cf = 0.01*(1-collision.unsqueeze(1))*(self.hidden2out_outcome_f(cf_h)+ y_cf[n][:,i:i+1]) + x_cf[n][:,i:i+1,-4]
                            else:
                                y_new_cf = self.hidden2out_outcome_f(cf_h) + y_cf[n][:,i:i+1]
                        if torch.sum(torch.isnan(y_new_cf))>0:
                            import pdb; pdb.set_trace()
                        cf_outcome_out[n].append(y_new_cf)
                    else:
                        cf_outcome_out[n].append(y_cf[n][:,i+1:i+2])

                    # roll out
                    if rollout and i >= burn_in -1:
                        x_cf[n][:,i+1] = tmp_pred_cf
                        y_cf[n][:,i+1] = y_new_cf.squeeze(1)
                    elif not rollout and i >= self.burn_in0:
                        x_cf_[n][:,i+1] = tmp_pred_cf
                    # VRNN update
                    x_1_cf = x_cf[n][:, i+1].clone()
                    x_1_cf_ = x_1_cf[:, :self.x_dim_permuted].reshape(batchSize,n_all_agents,n_feat)   
                    for ii in range(n_agents_):#+1):
                        if ii < n_agents:
                            if self.GraphNN:
                                _, h_micro_cf[ii][n] = self.gru_micro(torch.cat([x_1_cf_[:,ii], dec_in_cf[:,ii]], dim=1).reshape(batchSize,-1).unsqueeze(0), h_micro_cf[ii][n])#
                            else:
                                _, h_micro_cf[ii][n] = self.gru_micro(torch.cat([x_1_cf_[:,ii], dec_in_cf_], dim=1).reshape(batchSize,-1).unsqueeze(0), h_micro_cf[ii][n])#
                        elif ii == n_agents:
                            _, h_micro2_cf[n] = self.gru_micro2(torch.cat([x_1_cf[:,self.x_dim_permuted:], dec_in_cf_], dim=1).unsqueeze(0), h_micro2_cf[n])#
    

            # roll out
            if rollout and i >= burn_in-1 and i < len_time:
                x[:,i+1] = tmp_pred
                y[:,i+1] = y_new.squeeze(1)
            elif not rollout and i >= self.burn_in0:
                # print('mean factual collision: '+str(torch.mean(tmp_pred[:,-1])))
                x_[:,i+1] = tmp_pred
                if self.rollout_y_train and i >= burn_in -1:
                    y[:,i+1] = y_new.squeeze(1)
                    x[:,i+1] = tmp_pred
            # VRNN update----------------------------------     
            if i < len_time:   
                x_1 = x[:, i+1].clone()
                x_1_ = x_1[:, :self.x_dim_permuted].reshape(batchSize,n_all_agents,n_feat)   
                for ii in range(n_agents_):#+1
                    if ii < n_agents:
                        if self.GraphNN:
                            _, h_micro[ii] = self.gru_micro(torch.cat([x_1_[:,ii], dec_in[:,ii]], dim=1).reshape(batchSize,-1).unsqueeze(0), h_micro[ii])#
                        else:
                            _, h_micro[ii] = self.gru_micro(torch.cat([x_1_[:,ii], dec_in_], dim=1).reshape(batchSize,-1).unsqueeze(0), h_micro[ii])#
                    elif ii == n_agents:
                        _, h_micro2 = self.gru_micro2(torch.cat([x_1[:,self.x_dim_permuted:], dec_in_], dim=1).unsqueeze(0), h_micro2)#

        if CF_flag:
            x_out_cf = torch.stack(x_cf, dim=0) # if rollout else torch.stack(x_cf_, dim=0)
            x_out = x if rollout else x_
            cf_outcome_out = torch.stack(y_cf, dim=0)[:,:,1:].permute(0,2,1).unsqueeze(3)
        else:
            x_out = x if rollout else x_
            x_out_cf = torch.zeros(1,1).to(device)
 
        L_kl /= mean_time_length*n_agents 
        #if not Train and L_kl>1000:
        if torch.isnan(L_kl):
            import pdb; pdb.set_trace()
        return a_or_ipw_outputs, f_outcome_out, cf_outcome_out, f_h, L_kl, x_out, x_out_cf