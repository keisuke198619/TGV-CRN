import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from utils import GradientReversal
from utils import Attn, MLP, weights_init, edge2node, node2edge, encode_onehot 
from utils import batch_error, compute_x_ind, compute_collision, sample_gumbel_softmax
from utils import compute_global

class GDSW(nn.Module):
    def __init__(self, n_X_features, n_X_static_features, n_X_fr_types, n_Z_confounders,
                 attn_model, n_classes, args, hidden_size,
                 num_layers=2, dropout = 0.2):
        super().__init__()

        self.hidden_size = hidden_size
        self.batch_size = args.batch_size
        self.n_X_features = n_X_features
        self.n_X_static_features = n_X_static_features
        self.n_classes = n_classes
        self.num_layers = num_layers
        self.x_emb_size = 32
        self.dataset = args.data
        self.obs_w = args.observation_window # -1
        self.x_static_emb_size = n_X_static_features if n_X_static_features > 1 else 1
        self.attention = False # if "G" in args.model or "B" in args.model or "X" in args.model else True
        if self.dataset == 'carla': self.attention = False
        self.GraphNN = True if "G" in args.model else False
        self.balancing = True if "B" in args.model else False
        self.Pred_X = True if "X" in args.model else False
        self.theory = True if "T" in args.model else False
        self.theory2 = True if "TT" in args.model else False
        self.negativeGradient = True if "Ne" in args.model else False
        self.x_dim_predicted0 = args.x_dim_predicted
        self.x_dim_predicted = args.x_dim_predicted if self.theory else self.n_X_features
        self.vel = args.vel
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

        # added 
        if not self.attention:
            n_Z_confounders = 0
        
        self.z_dim = n_Z_confounders
        self.x_dim_permuted = args.x_dim_permuted    
        self.n_agents = args.n_agents
        self.n_dim_each_permuted = int(self.x_dim_permuted//self.n_agents)
        self.n_dim_each_pred = int(self.x_dim_predicted//self.n_agents)
        self.n_feat = self.n_dim_each_permuted 
        
        n_feat = self.n_feat
        self.xavier = True
        self.burn_in0 = args.burn_in0
        n_out = 8
        n_all_agents = self.n_agents
        self.n_X_unpermute = self.n_X_features - self.n_dim_each_permuted*n_all_agents
        if self.GraphNN:
            self.x_emb_graph_size = n_out*n_all_agents*(n_all_agents-1)
        else:
            self.x_emb_graph_size = 0
        
        self.n_t_classes = 1

        if self.attention:
            self.attn_f = Attn(attn_model, hidden_size)
            self.concat_f = nn.Linear(hidden_size * 2, hidden_size)
        if self.GraphNN:
            self.x2emb = nn.Linear(n_X_features-self.x_dim_permuted, self.x_emb_size)
            self.x_emb_size2 = 0
        else:
            self.x2emb = nn.Linear(n_X_features, self.x_emb_size)
            self.x_emb_size2 = self.x_emb_size
        # if n_X_static_features > 0:
        self.x_static2emb = nn.Linear(n_X_static_features, self.x_static_emb_size)

        # self.balancing or IPW
        if self.balancing and not self.negativeGradient:
            self.hidden2hidden_a_or_ipw = nn.Sequential(GradientReversal(),
                nn.Dropout(0.5),
                nn.Linear(self.x_emb_size2 + n_Z_confounders + self.x_static_emb_size + self.x_emb_graph_size, hidden_size),
                nn.Dropout(0.3),
                nn.ReLU(),
            )
        else:
            self.hidden2hidden_a_or_ipw = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(self.x_emb_size2 + n_Z_confounders + self.x_static_emb_size + self.x_emb_graph_size, hidden_size),
                nn.Dropout(0.3),
                nn.ReLU(),
            )
        self.hidden2out_a_or_ipw = nn.Linear(hidden_size, self.n_t_classes, bias=False)

        # Outcome
        self.hidden2hidden_outcome_f = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.x_emb_size + n_Z_confounders + self.x_static_emb_size + 1 + self.x_emb_graph_size, hidden_size),
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

        # RNN---------------------
        n_hid = 8
        self.z_dim_each = 4 
        n_layers = 2

        self.n_gnn = 1
        self.rnn_f = nn.GRUCell(input_size=self.x_emb_size + 1 + n_Z_confounders + self.x_emb_graph_size, hidden_size=hidden_size)
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

            for ped in range(self.n_gnn): # 0: prior, 1:encoder, 2:decoder
                if ped <= 1:
                    n_in = n_feat # +rnn_micro_dim #*n_all_agents # 
                elif ped == 2:
                    n_in = self.z_dim_each # +rnn_micro_dim 

                do_prob = 0

                self.mlp1[ped] = MLP(n_in, n_hid, n_hid, do_prob)
                self.mlp2[ped] = MLP(n_hid * 2 + self.n_edge_type, n_hid, n_hid, do_prob)

                self.mlp1[ped].apply(weights_init)
                self.mlp2[ped].apply(weights_init)

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

        # covariates
        if self.Pred_X:
            self.hidden2hidden_x = nn.Sequential( # nn.Dropout(0.5), nn.Dropout(0.3),
                nn.Linear(self.x_emb_size + n_Z_confounders + self.x_static_emb_size + self.x_emb_graph_size, hidden_size),
                nn.Softplus(),
                nn.ReLU(),
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

    def GNN(self, x, ped): # 0: prior, 1:encoder, 2:decoder
        rel_rec, rel_send = self.rel_rec, self.rel_send
        edge_type, node_type = self.edge_type, self.node_type
        batchSize = x.shape[0]
        if self.edge_type is not None:
            edge_type = edge_type.repeat(batchSize,1,1)
            node_type = node_type.repeat(batchSize,1,1)

        x = self.mlp1[ped](x)  # 2-layer ELU net per node # [128, 5, 8]

        x = node2edge(x, rel_rec, rel_send, edge_type) # [128, 20, 18]
        x = self.mlp2[ped](x)
        return x 

    def forward(self, x__, x_demo, f_treatment, y_, cf_treatment=None, target=None, burn_in=None, lengths=None):
        torch.autograd.set_detect_anomaly(True)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
        # added
        len_time = x__.size(1)-1
        batchSize = x__.size(0)
        n_all_agents = self.n_agents
        n_agents = self.n_agents
        n_feat = self.n_feat
        x_dim = self.n_dim_each_permuted
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
            burn_in = x__.size(1)-1
        else:
            rollout = True

        if self.theory: 
            ind_x, ind_x_, ind_x0, ind_x1 = compute_x_ind(self.x_dim_predicted,self.dataset,n_agents, vel=self.vel)
        else:
            ind_x, ind_x_, ind_x0, ind_x1 = compute_x_ind(self.x_dim_predicted0,self.dataset,n_agents, vel=self.vel)

        for i in range(len_time+horizon): # time
            x_0 = x[:, i].clone()
            x_1 = x[:, i+1].clone()

            if self.GraphNN: # graph embedding
                x_emb = self.x2emb(x[:, i, self.x_dim_permuted:])
                x__ = x[:, i, :self.x_dim_permuted].clone()
                x__ = x__.reshape(batchSize,n_all_agents,n_feat)
                x_emb_graph = self.GNN(x__, ped=0) .reshape(batchSize,-1)# [128, 20, 8]
                if self.attention:
                    f_zx_ = torch.cat((x_emb_graph, f_old), -1) # [128,225]
                    f_zx = torch.cat((x_emb_graph, x_emb, f_old), -1) 
                else:
                    f_zx_ = x_emb_graph
                    f_zx = torch.cat((x_emb_graph, x_emb), -1)
            else:
                x_emb = self.x2emb(x[:, i, :])
                if self.attention:
                    f_zx = torch.cat((x_emb, f_old), -1)
                else:
                    f_zx = x_emb
                f_zx_ = f_zx
                
            if i < len_time:
                # RNN
                f_zxs.append(f_zx_) # q in the paper
                try: f_inputs = torch.cat((f_zx, f_treatment[:,i].unsqueeze(1)), -1) # [128,225]
                except: import pdb; pdb.set_trace()
                f_hx = self.rnn_f(f_inputs, f_hx)
                
                f_outputs.append(f_hx)

                # attention
                if self.attention:
                    if i == 0:
                        f_concat_input = torch.cat((f_hx, f_hx), 1)
                    else:
                        f_attn_weights = self.attn_f(f_hx, torch.stack(f_outputs))
                        f_context = f_attn_weights.bmm(torch.stack(f_outputs).transpose(0, 1))
                        f_context = f_context.squeeze(1)
                        f_concat_input = torch.cat((f_hx, f_context), 1)

                    f_concat_output = torch.tanh(self.concat_f(f_concat_input))
                    f_old = f_concat_output
                
                # Y
                f_hidden = torch.cat((f_zx, x_demo_emd), -1)
                f_h_ = self.hidden2hidden_outcome_f(torch.cat((f_hidden, f_treatment[:,i:i+1]), -1))
                f_h.append(f_h_)

                if self.Pred_X:
                    f_x_ = self.hidden2hidden_x(f_hidden) 
                    if self.x_residual: 
                        f_x_ = self.hidden2out_x(f_x_)
                        x_new = torch.zeros(batchSize,self.x_dim_predicted).to(device)
                        x_new[:,ind_x] = f_x_[:,ind_x] + x_0[:, ind_x0]
                        x_new[:,ind_x_] = f_x_[:,ind_x_] 
                        if torch.mean(torch.abs(x_new)) > 50:
                            import pdb; pdb.set_trace()
                    else:
                        x_new = self.hidden2out_x(f_x_) 

                    # compute global parameters
                    if i >= self.burn_in0:
                        tmp_pred = compute_global(x_new,x[:,i],x[:,i-1],x[:,i+1],f_treatment[:,i:i+1],x_demo,device,i,self.self_1,self.self_2,CF_flag=False)
                        if self.theory and 'carla' in self.dataset: # collision
                            input_collision = x[:,i] # tmp_pred
                            collision = self.compute_collisions(input_collision,batchSize,device,x_dim,n_agents) 
                            tmp_pred[:,-1] = collision
                # outcome 
                if i >= self.burn_in0:
                    if i == 0 or not self.y_residual:
                        y_new = self.hidden2out_outcome_f(f_h_)
                    else:
                        if self.theory and 'carla' in self.dataset and self.Pred_X: 
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
                    if self.GraphNN:
                        x_emb_cf = self.x2emb(x_0_cf[:, self.x_dim_permuted:])
                        x_cf_ = x_0_cf[:, :self.x_dim_permuted]
                        x_cf_ = x_cf_.reshape(batchSize,n_all_agents,n_feat)
                        x_emb_graph_cf = self.GNN(x_cf_, ped=0) .reshape(batchSize,-1)# [128, 20, 8]
                        if self.attention:
                            cf_zx = torch.cat((x_emb_graph_cf, x_emb_cf, cf_old[n]), -1)
                        else:
                            cf_zx = torch.cat((x_emb_graph_cf, x_emb_cf), -1)
                    else:
                        x_emb_cf = self.x2emb(x_0_cf)
                        if self.attention:
                            cf_zx = torch.cat((x_emb_cf, cf_old[n]), -1)
                        else:
                            cf_zx = x_emb_cf

                    cf_inputs = torch.cat((cf_zx, cf_treatment[:,i:i+1,n]), -1)

                    # RNN
                    cf_hx[n] = self.rnn_f(cf_inputs, cf_hx[n])
                    cf_outputs[n].append(cf_hx[n])
                    # attention
                    if self.attention:
                        if i == 0:
                            cf_concat_input = torch.cat((cf_hx[n], cf_hx[n]), 1)
                        else:
                            cf_attn_weights = self.attn_f(cf_hx[n], torch.stack(cf_outputs[n]))
                            cf_context = cf_attn_weights.bmm(torch.stack(cf_outputs[n]).transpose(0, 1))
                            cf_context = cf_context.squeeze(1)
                            cf_concat_input = torch.cat((cf_hx[n], cf_context), 1)

                        cf_concat_output = torch.tanh(self.concat_f(cf_concat_input))
                        cf_old[n] = cf_concat_output
                    
                    # added
                    cf_hidden = torch.cat((cf_zx, x_demo_emd), -1)
                    cf_h = self.hidden2hidden_outcome_f(torch.cat((cf_hidden, cf_treatment[:,i:i+1,n]), -1))

                    if self.Pred_X:
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

                    if i >= self.burn_in0:
                        if i == 0 or not self.y_residual:
                            y_new_cf = self.hidden2out_outcome_f(cf_h) 
                            if 'nba' in self.dataset: 
                                y_new_cf = torch.clamp(y_new_cf,max=1)
                        else:
                            if self.theory and 'carla' in self.dataset and self.Pred_X: 
                                y_new_cf = 0.01*(1-collision.unsqueeze(1))*self.hidden2out_outcome_f(cf_h) + x_cf[n][:,i:i+1,-4] #y_cf[n][:,i:i+1] # 0->1, 1->0
                            else:
                                y_new_cf = self.hidden2out_outcome_f(cf_h) + y_cf[n][:,i:i+1]
                        cf_outcome_out[n].append(y_new_cf)
                    else:
                        cf_outcome_out[n].append(y_cf[n][:,i+1:i+2])

                    # roll out
                    if rollout and i >= burn_in -1:
                        y_cf[n][:,i+1] = y_new_cf.squeeze(1)
                        if self.Pred_X:
                            x_cf[n][:,i+1] = tmp_pred_cf
                    elif self.Pred_X and not rollout and i >= self.burn_in0:
                        x_cf_[n][:,i+1] = tmp_pred_cf
            # roll out
            if rollout and i >= burn_in-1 and i < len_time:
                if self.Pred_X:
                    x[:,i+1] = tmp_pred
                y[:,i+1] = y_new.squeeze(1)
            elif self.Pred_X and not rollout and i >= self.burn_in0:
                if self.Pred_X:
                    x_[:,i+1] = tmp_pred
                if self.rollout_y_train and i >= burn_in -1:
                    y[:,i+1] = y_new.squeeze(1)
                    if self.Pred_X:
                        x[:,i+1] = tmp_pred

        if CF_flag: 
            cf_outcome_out = torch.stack(y_cf, dim=0)[:,:,1:].permute(0,2,1).unsqueeze(3)

        a_or_ipw_outputs = []
        for i in range(len(f_zxs)): # time
            h = torch.cat((f_zxs[i], x_demo_emd), -1)
            h = self.hidden2hidden_a_or_ipw(h)
            out = self.hidden2out_a_or_ipw(h)
            a_or_ipw_outputs.append(out)

        if self.Pred_X:
            if CF_flag:
                x_out_cf = torch.stack(x_cf, dim=0) # if rollout else torch.stack(x_cf_, dim=0)
                x_out = x if rollout else x_
            else:
                x_out = x if rollout else x_
                x_out_cf = torch.zeros(1,1).to(device)
        else:
            x_out = torch.zeros(1).to(device) # dummy
            x_out_cf = torch.zeros(1,1).to(device) # dummy

        return a_or_ipw_outputs, f_outcome_out, cf_outcome_out, f_h, x_out, x_out_cf
