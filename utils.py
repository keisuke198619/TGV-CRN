from PIL import Image
import math
import numpy as np
import torch
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F

class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat','concat2']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

        elif self.method == 'concat2':
            self.attn = nn.Linear(self.hidden_size * 3, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def concat_score2(self, hidden, encoder_output):
        h = torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)
        h = torch.cat((h, hidden*encoder_output),2)
        energy = self.attn(h).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)
        elif self.method == 'concat2':
            attn_energies = self.concat_score2(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

class MLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]
        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        return self.batch_norm(x)

def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad

def weights_init(m):
    # https://discuss.pytorch.org/t/weight-initialization-with-a-custom-method-in-nn-sequential/24846
    # https://blog.snowhork.com/2018/11/pytorch-initialize-weight
    # for mm in range(len(m)):
    mm=0
    if type(m) == nn.Linear: # in ,nn.GRU
        nn.init.xavier_normal_(m.weight)
    elif type(m) == nn.GRU:
        nn.init.xavier_normal_(m.weight_hh_l0)
        nn.init.xavier_normal_(m.weight_ih_l0)
######################################################

def edge2node(x, rel_rec, rel_send, node_type):
    # x: (batch,agents*(agents-1),hidden)
    # rel_rec: (agents*(agents-1),agents)
    # incoming: (batch,agents,hidden)
    # node_type: (batch,agents,2)
    incoming = torch.matmul(rel_rec.t(), x)
    if node_type is not None:
        incoming = torch.cat([incoming, node_type], dim=2)
    return incoming / incoming.size(1)

def node2edge(x, rel_rec, rel_send, edge_type):
    # x: (batch,agents,hidden)
    # rel_rec,rel_send: (agents*(agents-1),agents)
    # edge_type: (batch,agents*(agents-1),2)
    receivers = torch.matmul(rel_rec, x) # (batch,agents*(agents-1),hidden)
    senders = torch.matmul(rel_send, x) # (batch,agents*(agents-1),hidden)
    if edge_type is not None:
        edges = torch.cat([senders, receivers, edge_type], dim=2)
    else:
        edges = torch.cat([senders, receivers], dim=2)
    return edges

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                            dtype=np.int32)
    return labels_onehot


##############################################################

class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)

    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

def cudafy_list(states):
    for i in range(len(states)):
        states[i] = states[i].cuda()
    return states

######################################################################
############################ MODEL UTILS #############################
######################################################################

def sample_gauss(mean, std):
    eps = torch.FloatTensor(std.size()).normal_()
    if mean.is_cuda:
        eps = eps.cuda()
    return eps.mul(std).add_(mean)

######################################################################
############################## GAUSSIAN ##############################
######################################################################

def nll_gauss(mean, std, x, pow=False,Sum=True):
    pi = torch.FloatTensor([math.pi])
    if mean.is_cuda:
        pi = pi.cuda()
    if not pow:
        nll_element = (x - mean).pow(2) / std.pow(2) + 2*torch.log(std) + torch.log(2*pi)
    else:
        nll_element = (x - mean).pow(2) / std + torch.log(std) + torch.log(2*pi)

    nll = 0.5 * torch.sum(nll_element) if Sum else 0.5 * torch.sum(nll_element,1)
    return nll
    

def kld_gauss(mean_1, std_1, mean_2, std_2, Sum=True, index=None):
    kld_element =  (2 * torch.log(std_2+1e-3) - 2 * torch.log(std_1+1e-3) + 
        (std_1.pow(2) + (mean_1 - mean_2).pow(2)) / (std_2.pow(2)+1e-3) - 1)
    # kld = 0.5 * torch.sum(kld_element) if Sum else 0.5 * torch.sum(kld_element,1)
    if index is not None and torch.sum(index) < len(index):
        kld_element[~index] = 0
    if Sum:
        kld = torch.sum(kld_element)  
    else:
        kld = torch.sum(kld_element,1)    
    if torch.isnan(kld): # kld > 1000000 or 
        import pdb; pdb.set_trace()
    return 0.5 *kld

def batch_error(predict, true, Sum=True, sqrt=False, diff=True, index=None):#, normalize=False):
    # error = torch.sum(torch.sum((predict[:,:2] - true[:,:2]),1))
    # index = (true[:,0]>9998)
    if predict.shape[1] > 1:
        if sqrt:
            # error = torch.sqrt(error)
            if diff:
                error = torch.norm(torch.abs(predict-true)+1e-6,p=2,dim=1)
            else:
                error = torch.norm(predict.abs()+1e-6,p=2,dim=1)
        else:
            if diff:
                error = torch.sum((torch.abs(predict - true)+1e-6).pow(2),1) # [:,:2]
            else:
                error = torch.sum((predict.abs()+1e-6).pow(2),1) # [:,:2]
    else:
        error = torch.abs(predict - true) 
    
    if index is not None and torch.sum(index) < len(index):
        error[~index] = 0

    if Sum:
        error = torch.sum(error)
    return error

def batch_error_angle(predict, true, Sum=True, index=None):
    # https://discuss.pytorch.org/t/custom-loss-function-for-discontinuous-angle-calculation/58579/2
    a = predict - true
    error = (a + torch.pi) % (2*torch.pi) - torch.pi
    error = torch.abs(a)

    if index is not None and torch.sum(index) < len(index):
        error[~index] = 0

    if Sum:
        error = torch.sum(error)
    return error
######################################################################
############### sample_gumbel_softmax ################################
######################################################################

def sample_gumbel(shape, eps=1e-20):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    unif = torch.rand(*shape).to(device)
    g = -torch.log(-torch.log(unif + eps))
    return g

def sample_gumbel_softmax(logits, temperature):
    """
        Input:
        logits: Tensor of log probs, shape = BS x k
        temperature = scalar
        
        Output: Tensor of values sampled from Gumbel softmax.
                These will tend towards a one-hot representation in the limit of temp -> 0
                shape = BS x k
    """
    g = sample_gumbel(logits.shape)
    h_ = (g + logits)/temperature
    h_max = h_.max(dim=-1, keepdim=True)[0]
    h_ = h_ - h_max
    cache = torch.exp(h_)
    y = cache / cache.sum(dim=-1, keepdim=True)
    return y
######################################################################
############################## collision ##############################
######################################################################
def compute_collision(pred,x_dim,n_agents,k):
    extra_dim = 4
    ego_pos = pred[:,:-extra_dim].reshape(-1,n_agents,x_dim)[:,0,1:3] 
    other_pos = pred[:,:-extra_dim].reshape(-1,n_agents,x_dim)[:,k,1:3]
    ego_size = pred[:,:-extra_dim].reshape(-1,n_agents,x_dim)[:,0,5:7] 
    other_size = pred[:,:-extra_dim].reshape(-1,n_agents,x_dim)[:,k,5:7]
    tmp_dist = torch.norm(ego_pos-other_pos,p=2,dim=1)
    if torch.sum(torch.isnan(tmp_dist)):
        import pdb; pdb.set_trace()
    # tmp_dist = 1/(1e-6+torch.norm(ego_pos-other_pos,p=2,dim=1))
    return torch.cat([tmp_dist.unsqueeze(1),ego_size,other_size],1)

######################################################################
############################## x_residual ##############################
######################################################################
def compute_x_ind(x_dim_predicted,dataset,n_agents,vel=True):
    # ind_x: residual dimension
    # ind_x_: not residual dimension
    # ind_x0: residual dimension in all dimension
    # ind_x1: predicted dimension
    ind_x_all = torch.arange(x_dim_predicted)
    if dataset == 'nba': # or 'boid' in dataset:
        if vel:
            #x_dim = 5
            #ind_x1 = torch.cat([torch.arange(2,n_agents*x_dim,x_dim),torch.arange(3,n_agents*x_dim,x_dim),torch.arange(4,n_agents*x_dim,x_dim)],0) 
            x_dim = 4
            ind_x1 = torch.cat([torch.arange(0,n_agents*x_dim,x_dim),torch.arange(1,n_agents*x_dim,x_dim)],0) 
            ind_x = torch.cat([torch.arange(0,n_agents*x_dim,x_dim),torch.arange(1,n_agents*x_dim,x_dim)],0) 
            ind_x0 = torch.cat([torch.arange(0,n_agents*x_dim,x_dim),torch.arange(1,n_agents*x_dim,x_dim)],0) 
        else:
            x_dim = 2 
            ind_x = ind_x_all
            ind_x1 = ind_x_all
            ind_x0 = ind_x_all
    elif dataset == 'carla':
        x_dim = 7
        ind_x = torch.cat([torch.arange(1,n_agents*x_dim,x_dim),torch.arange(2,n_agents*x_dim,x_dim)],0)
        ind_x1 = torch.cat([torch.arange(0,n_agents*x_dim,x_dim),torch.arange(1,n_agents*x_dim,x_dim),torch.arange(2,n_agents*5,5)],0)
        ind_x0 = torch.cat([torch.arange(1,n_agents*x_dim,x_dim),torch.arange(2,n_agents*x_dim,x_dim)],0)
    elif 'boid' in dataset:
        x_dim = 7 
        ind_x1 = torch.cat([torch.arange(2,n_agents*x_dim,x_dim),torch.arange(3,n_agents*x_dim,x_dim)],0) # torch.arange(2,n_agents*x_dim,x_dim) # 
        ind_x = torch.arange(0) # torch.arange(0,n_agents*2,2)
        ind_x0 = torch.arange(0) # torch.arange(2,n_agents*x_dim,x_dim)
    else: 
        ind_x = ind_x_all
        ind_x1 = ind_x_all
        ind_x0 = ind_x_all
    combined = torch.cat((ind_x_all, ind_x))
    uniques, counts = combined.unique(return_counts=True)
    ind_x_ = uniques[counts == 1] # setdiff  
    ind_x1 = ind_x1.reshape(int(x_dim_predicted//n_agents),n_agents) # dim,agents
    return ind_x, ind_x_, ind_x0, ind_x1 # , ind_x1_,

######################################################################
########################## MISCELLANEOUS #############################
######################################################################
def std_ste(x,std=False):
    if std:
        return np.std(x)
    else:
        return np.std(x)/np.sqrt(len(x))

######################################################################
########################## ROLLOUT ###################################
######################################################################

# ===================================================================
def categorize_dist_def_feet(distance):
    c_dist_def_feet = distance.clone()
    feet_m = 0.3048
    c_dist_def_feet[distance < 2*feet_m] = 0 
    c_dist_def_feet[(distance >= 2*feet_m) * (distance < 4*feet_m)] = 1
    c_dist_def_feet[(distance >= 4*feet_m) * (distance < 6*feet_m)] = 2
    c_dist_def_feet[distance >= 6*feet_m] = 3 
    return c_dist_def_feet.to(torch.int64)
# ===================================================================
def categorize_shot_area(distance_goal,ball_pos,device):
    # ['shot_area_name']  = ['RESTRICTED AREA','IN THE PAINT','MID-RANGE','CORNER 3','ABOVE THE BREAK 3']
    # https://www.thebirdwrites.com/2018/12/29/18160043/anthony-davis-new-career-high-points-paint-new-orleans-pelicans-dallas-mavericks-luka-doncic
    # https://www.recunlimited.com/blog/diagrams-basketball-courts/
    feet_m = 0.3048
    ring23p = 23*feet_m + 0.2286 # 23ft 9in
    out_area = torch.zeros(distance_goal.shape[0],3).to(torch.int64).to(device)
    out_shot2or3 = torch.zeros(distance_goal.shape[0]).to(torch.int64).to(device)

    # RESTRICTED AREA
    out_area[distance_goal <= 8*feet_m,0] = 1 
    out_shot2or3[distance_goal <= 16*feet_m] = 0
    # IN THE PAINT
    out_area[distance_goal**2 <= (15*feet_m-0.381)**2+(8*feet_m)**2,1] = 1 
    out_shot2or3[distance_goal**2 <= (15*feet_m-0.381)**2+(8*feet_m)**2] = 0   
    # CORNER 3
    cond1 = ball_pos[:,0] <= 14*feet_m
    cond2 = ball_pos[:,1] <= 3*feet_m
    cond3 = ball_pos[:,1] >= 47*feet_m
    out_area[cond1*cond2,2] = 1 
    out_area[cond1*cond3,2] = 1 
    out_shot2or3[cond1*cond2] = 1
    out_shot2or3[cond1*cond3] = 1
    # ABOVE THE BREAK 3
    out_area[distance_goal >= ring23p,2] = 1 
    out_shot2or3[distance_goal >= ring23p] = 1
    # other: MID-RANGE 

    return out_shot2or3, out_area

def calc_feature_nba(Pballxy,Pballdist,Fs,i,max_v,L_Ring,Ball_Hold,posxy_,posxy_prev,x_prev,vel_prev,vel_prev2,vel_pred,x_demo,Ball_OF,device,ShotClock,Clock,batchSize,pass_speed,time2catch_ballOF,defender_speed):
    posxy = posxy_.clone()

    # Distances and angles (with respect to the hoop) between players
    dist_B_def = torch.sqrt(torch.sum((Pballxy.unsqueeze(1).repeat(1,5,1)-posxy[:,5:10,:])**2,2))
    dist_B_def_index = torch.argsort(dist_B_def)
    dist_B_def = torch.diagonal(dist_B_def[:,dist_B_def_index[:,0]])
    dist_def_feet = categorize_dist_def_feet(dist_B_def- time2catch_ballOF*defender_speed*max_v)
    
    # shot probability 
    shot_prob_all = x_demo[:,11:].reshape(-1,5,4,2) # batch,players,dist,2/3p
    shot2or3, shot_area = categorize_shot_area(Pballdist,Pballxy,device)
    
    shot_prob2 = torch.diagonal(shot_prob_all[:,Ball_OF],dim1=0,dim2=1).permute(2,0,1)
    shot_prob2 = torch.diagonal(shot_prob2[:,dist_def_feet],dim1=0,dim2=1).permute(1,0)
    shot_prob2 = torch.diagonal(shot_prob2[:,shot2or3])

    # adjust such that the probability is reduced by 0.2 at 12.73m.
    # 25ft -> 34 ft -> 43ft
    feet_m = 0.3048
    ring23p = 23*feet_m + 0.2286 # 23ft 9in
    ring2half = 47*feet_m - 4*feet_m - 0.381
    ring2half2 = (ring2half + 25*feet_m)/2
    shot_prob2[(shot2or3)&(Pballdist > ring2half2)] = shot_prob2[(shot2or3)&(Pballdist > ring2half2)] - 0.1
    shot_prob2[(shot2or3)&(Pballdist > ring2half)] = shot_prob2[(shot2or3)&(Pballdist > ring2half)] - 0.1
    # shot_prob2[shot2or3] = shot_prob2[shot2or3] - 0.3/(ring2half-ring23p)*(Pballdist-ring23p)
    shot_prob2[shot2or3] = torch.clamp(shot_prob2[shot2or3],min=0)
    # y: shot probablitiy when pass or shot (possible maximal value)
    # we assume that the pass speed is constant (90 percentile of the actual),
    # the nearest defender speed is constant (95 percentile of the actual),
    # and ball reciever does not move
    
    min_clock = torch.min(torch.cat([ShotClock.unsqueeze(1)*24,Clock.unsqueeze(1)*720],1),1)[0]
    other = torch.arange(5)
    others = torch.zeros((batchSize,4)).to(torch.int64).to(device) # torch.arange(5).repeat(batchSize,1)
    for b in range(batchSize):
        others[b] = torch.cat([other[:Ball_OF[b]].unsqueeze(0),other[Ball_OF[b]+1:].unsqueeze(0)],1)
    # others = torch.cat([others[:,:Ball_OF],others[:,Ball_OF+1:]],1)
    ind_other = torch.zeros(4).to(torch.int64).to(device)
    # shot_prob_others = torch.zeros(4,4,2).to(device)
    shot_prob_other = torch.zeros(batchSize,4).to(device)
    time2catch = torch.zeros(batchSize,4).to(device)
    # next_ballxys = torch.zeros(batchSize,4,2).to(device)
    
    dist_def_others_feet = torch.zeros((batchSize,4)).to(torch.int64).to(device)
    shot2or3_other = torch.zeros((batchSize,4)).to(torch.int64).to(device)
    # other_xy_new = torch.zeros((batchSize,4,2)).to(torch.int64).to(device)
    # ballxy = posxy[:,10,:]
    ballxy_prev = posxy_prev[:,10,:]
    for p in range(4):             
        other_xy = torch.diagonal(posxy[:,others[:,p],:],dim1=0,dim2=1).transpose(1,0)
        dist_other_def = torch.sqrt(torch.sum((other_xy.unsqueeze(1).repeat(1,5,1)-posxy[:,5:10,:])**2,2))
        dist_other_def_index = torch.argsort(dist_other_def)
        dist_other_def = torch.diagonal(dist_other_def[:,dist_other_def_index[:,0]]) 
        ball_other = other_xy-ballxy_prev
        dist_B_other = torch.sqrt(torch.sum(ball_other**2,1))
        time2catch[:,p] = dist_B_other/pass_speed/max_v + time2catch_ballOF

        defender_closeness = dist_other_def-time2catch[:,p]*defender_speed*max_v

        dist_def_others_feet[:,p] = categorize_dist_def_feet(defender_closeness)
        vec_B_other = other_xy - L_Ring.repeat(batchSize,1)
        shot2or3_other[:,p], _ = categorize_shot_area(torch.sqrt(torch.sum(vec_B_other**2,1)),other_xy,device) # vec_B_other
        shot_prob_others = torch.diagonal(shot_prob_all[:,others[:,p]],dim1=0,dim2=1).permute(2,0,1)
        shot_prob_others = torch.diagonal(shot_prob_others[:,dist_def_others_feet[:,p]],dim1=0,dim2=1).permute(1,0)
        shot_prob_other[:,p] = torch.diagonal(shot_prob_others[:,shot2or3_other[:,p]])

        shot_prob_other[time2catch[:,p] > min_clock,p] = 0
        # shot_prob_other[Ball_Hold == 0,p] = 0
    #if torch.sum(shot_prob_other)==0:
    #    import pdb; pdb.set_trace()

    return dist_def_feet, shot2or3, shot_area, shot_prob2, shot_prob_other, dist_def_others_feet, time2catch, shot2or3_other, others, posxy # , other_xy_new 


def compute_global(x_pred,x_prev,x_prev2,x,treatment,x_demo,device,i,self_1,self_2,CF_flag=False):
    batchSize = x.shape[0]
    self_dataset, self_n_dim_each_permuted, self_vel, self_n_agents, self_theory, self_x_dim_predicted, self_x_dim_permuted = self_1
    if self_dataset == 'nba':
        self_theory2, self_max_v = self_2
    elif 'carla' in self_dataset:
        self_max_v, self_max_p, self_max_y = self_2
    elif 'boid' in self_dataset:
        self_max_p = self_2

    if self_dataset == 'nba':
        x_ = x.clone()
        Fs = 5
        dim = self_n_dim_each_permuted
        out = x_
        if self_vel:
            # t-1
            x_prev_ = x_prev[:,:self_n_agents*dim].reshape(-1,self_n_agents,dim)
            # 
            posxy_prev = x_prev_[:,:,:2]
            vel_prev = x_prev_[:,:,2:]
            # t-2
            x_prev2_ = x_prev2[:,:self_n_agents*dim].reshape(-1,self_n_agents,dim)
            vel_prev2 = x_prev_[:,:,2:]
            if dim==5:
                vel_prev_ = vel_prev[:,:,:2]*vel_prev[:,:,2].unsqueeze(2).repeat(1,1,2)
                vel_prev2_ = vel_prev2[:,:,:2]*vel_prev2[:,:,2].unsqueeze(2).repeat(1,1,2)
            
            if self_theory:
                vel_pred = x_pred[:,:self_x_dim_predicted-1].reshape(-1,self_n_agents,dim-2)
            else:
                vel_pred = x_pred[:,:self_n_agents*dim].reshape(-1,self_n_agents,dim)[:,:,2:4]
                # vel_pred = x_pred[:,self_n_agents*2:self_n_agents*dim].reshape(-1,self_n_agents,dim-2)#dim)[:,:,2:5]
            
            if dim==5:
                vel_pred_norm = torch.exp(torch.clamp(vel_pred[:,:,2],max=2))
                vel_pred_cs = torch.clamp(vel_pred[:,:,:2],min=-1,max=1)
                vel_pred_ = vel_pred_cs*vel_pred_norm.unsqueeze(2).repeat(1,1,2)
                # Integration using Simpson's rule
                posxy = posxy_prev + (vel_prev2_+4*vel_prev_+vel_pred_)/6/Fs*self_max_v # vel_prev_/Fs # 
                vel_pred = torch.cat([vel_pred_cs,vel_pred_norm.unsqueeze(2)],2)
            else:
                vel_pred = torch.clamp(vel_pred,min=-7/self_max_v,max=7/self_max_v)
                # Integration using Simpson's rule
                posxy = posxy_prev + (vel_prev2+4*vel_prev+vel_pred)/6/Fs*self_max_v 

            if not self_theory:
                out = x_pred
                #out[:,:self_x_dim_permuted] = torch.cat([posxy,vel_pred],2).reshape(-1,self_n_agents*dim)
                #out[:,self_x_dim_permuted:] = x_pred[:,self_x_dim_permuted:]
        else:
            x_[:, :self_x_dim_predicted] = x_pred
            pos = x_pred[:,:self_x_dim_permuted]
            posxy = pos[:,:self_n_agents*2].reshape((-1,self_n_agents,2))
            posxy_prev = x_prev[:,:self_n_agents*2].reshape((-1,self_n_agents,2))
            vel_prev, vel_prev2, vel_pred = None, None, None

        if self_theory:
            feet_m = 0.3048
            sixfeet = 6*feet_m
            pass_speed = 12/self_max_v
            defender_speed = 4.4/self_max_v
            L_Ring = torch.tensor([1.4478,7.6200])
            
            # feat = pos_[:,:22], restricted,paint,3p,dist_def_feet,dist_def_others_feet,
            # shot_prob,shot_prob_other_max,Clock,ShotClock,Ball_OF_,Ball_Hold_
            # static = [player_ID,shot_prob_all]
            player_ID = x_demo[1:11]

            # Game time and quarter time left on the clock
            Clock_prev = 720*x_prev[:,self_x_dim_permuted+7]
            Clock = (Clock_prev - 1/Fs)/720
            ShotClock_prev = 24*x_prev[:,self_x_dim_permuted+8]
            ShotClock = (ShotClock_prev - 1/Fs)/24
            
            Ball_OF_prev = x_prev[:,self_x_dim_permuted+9].to(torch.int64)
            Ball_Hold_prev = x_prev[:,self_x_dim_permuted+10].to(torch.int64)
            Ball_OF = Ball_OF_prev.clone()
            Ball_Hold = Ball_Hold_prev.clone()

            ballxy = posxy[:,-1]
            ballxy_true = x[:,self_x_dim_permuted-2:self_x_dim_permuted]
            ballxy_prev = posxy_prev[:,-1,:]

            ind_treat_1 = torch.where(treatment==1)[0]
            # ind_treat_0 = torch.where(treatment==0)
            
            # elif treatment==0:
            Pballxy = torch.diagonal(posxy[:,Ball_OF,:],dim1=0,dim2=1).transpose(1,0)
            Pballxy_prev = torch.diagonal(posxy_prev[:,Ball_OF,:],dim1=0,dim2=1).transpose(1,0)
            vec_B = Pballxy - L_Ring
            Pballdist = torch.sqrt(torch.sum(vec_B**2,dim=1))

            # compute time to catch if no ball hold
            time2catch_ballOF = torch.zeros(batchSize).to(device)
            next_ballxy = ballxy_prev.clone()
            if self_vel:
                if dim==5:
                    # ballvel_prev = torch.cat([(vel_prev[:,-1,2]*vel_prev[:,-1,0]).unsqueeze(1),(vel_prev[:,-1,2]*vel_prev[:,-1,1]).unsqueeze(1)],1)#.reshape(-1,self_n_agents,2)
                    ballvel_prev = vel_prev[:,-1,:2]*vel_prev[:,-1,2].unsqueeze(1).repeat(1,2)
                else:
                    ballvel_prev = vel_prev[:,-1,:2]
                next_ballvel = ballvel_prev.clone()

            ball_Pball = Pballxy-ballxy_prev
            dist_B_ballOF = torch.sqrt(torch.sum(ball_Pball**2,1))+1e-4
            
            # pass 
            if self_theory2:
                if torch.sum(Ball_Hold == 1)>0:
                    next_ballxy[Ball_Hold == 1] = (ballxy_prev + Pballxy-Pballxy_prev)[Ball_Hold == 1]

                if torch.sum(Ball_Hold == 0)>0: # no hold 
                    if True:
                        if self_vel:
                            ind_Ball_Hold_0 = torch.where(Ball_Hold == 0)[0]
                            next_ballvel[ind_Ball_Hold_0] = ballvel_prev[ind_Ball_Hold_0]
                            next_ballxy[ind_Ball_Hold_0] = ballxy_prev[ind_Ball_Hold_0] + ballvel_prev[ind_Ball_Hold_0]/Fs*self_max_v
                        else:
                            next_ballxy[Ball_Hold == 0] = (ballxy_prev + ball_Pball/dist_B_ballOF.unsqueeze(1).repeat(1,2)*(pass_speed*self_max_v/Fs))[Ball_Hold == 0]
                    else:
                        time2catch_ballOF[Ball_Hold == 0] = dist_B_ballOF[Ball_Hold == 0]/pass_speed/self_max_v
                        zeroindex = torch.where((Ball_Hold == 0)&(torch.abs(time2catch_ballOF)<=1e-2))[0]
                        nonzeroindex = torch.where((Ball_Hold == 0)&(torch.abs(time2catch_ballOF)>1e-2))[0]
                        next_ballxy[nonzeroindex] = (ballxy_prev + ball_Pball*(1/Fs)/time2catch_ballOF.unsqueeze(1).repeat(1,2))[nonzeroindex]

            else:
                next_ballxy = ballxy
            if torch.sum(Ball_Hold == 0)>0:
                # new ball hold
                ind_newhold = (dist_B_ballOF <= 1.5) & (Ball_Hold == 0)
                Ball_Hold[ind_newhold] = 1 
                next_ballxy[ind_newhold] = (ballxy_prev + ball_Pball*0.5)[ind_newhold]

            posxy[:,10] = next_ballxy
            if self_vel and self_theory2:
                if dim==5:
                    next_ballvel_norm = torch.norm(next_ballvel+1e-4,dim=1).unsqueeze(1)
                    if torch.sum(torch.isinf(torch.abs(next_ballvel_norm)))>0 or torch.sum(torch.isnan(next_ballvel_norm))>0:
                        import pdb; pdb.set_trace()
                    nonzeroindex = torch.where(next_ballvel_norm>1e-3)[0]
                    next_ballvel_ = next_ballvel[nonzeroindex]/next_ballvel_norm[nonzeroindex].repeat(1,2)
                    vel_pred[nonzeroindex,10] = torch.cat([next_ballvel_,next_ballvel_norm[nonzeroindex]],1)
                else:
                    vel_pred[:,10] = next_ballvel

            dist_def_feet, shot2or3, shot_area, shot_prob2, shot_prob_other, dist_def_others_feet, time2catch, shot2or3_other, others, _ = calc_feature_nba(
                Pballxy,Pballdist,Fs,i,self_max_v,L_Ring,Ball_Hold,posxy,posxy_prev,x_prev,vel_prev,vel_prev2,vel_pred,x_demo,Ball_OF,device,ShotClock,Clock,batchSize,pass_speed,time2catch_ballOF,defender_speed)

            shot_prob_other_max = torch.max(shot_prob_other,1)[0].unsqueeze(1)
            dist_def_others_feet_max = torch.diagonal(dist_def_others_feet[:,torch.max(shot_prob_other,1)[1]]) # torch.max(dist_def_others_feet,1)[0].unsqueeze(1)

            # if treatment==1
            # if Ball_Hold == 0, treatment cannot be performed (such situation is eliminated)
            ind_treat_2 = ind_treat_1 # torch.where((Ball_Hold == 0)&(treatment[:,0]==0))[0] # ind_treat_1 #  
            if CF_flag and torch.sum(ind_treat_2) > 0: # 
                next_ballxy_ = next_ballxy.clone()
                if self_vel:
                    next_ballvel_ = next_ballvel.clone()
                
                Ball_Hold[ind_treat_2] = 0 
                best_shot_prob_player = torch.max(shot_prob_other,1)[1]
                Ball_OF[ind_treat_2] = torch.diagonal(others[:,best_shot_prob_player[ind_treat_2]]) 
                time2catch_ballOF[ind_treat_2] = time2catch[ind_treat_2,best_shot_prob_player[ind_treat_2]] # torch.diagonal()
                
                Pballxy = torch.diagonal(posxy[:,Ball_OF,:],dim1=0,dim2=1).transpose(1,0)
                vec_B = Pballxy - L_Ring
                Pballdist = torch.sqrt(torch.sum(vec_B**2,dim=1))+1e-4

                dist_def_feet, shot2or3, shot_area, shot_prob2, shot_prob_other, dist_def_others_feet, _, _, _, posxy = calc_feature_nba(
                    Pballxy,Pballdist,Fs,i,self_max_v,L_Ring,Ball_Hold,posxy,posxy_prev,x_prev,vel_prev,vel_prev2,vel_pred,x_demo,Ball_OF,device,ShotClock,Clock,batchSize,pass_speed,time2catch_ballOF,defender_speed)
                shot_prob_other_max[ind_treat_2] = torch.max(shot_prob_other[ind_treat_2],1)[0].unsqueeze(1)
                dist_def_others_feet_max = torch.diagonal(dist_def_others_feet[:,torch.max(shot_prob_other,1)[1]]) # torch.max(dist_def_others_feet,1)[0].unsqueeze(1)
                #ind_max = torch.max(shot_prob_other,1)[1]
                #dist_def_others_feet_max[ind_treat_2] = torch.diagonal(dist_def_others_feet[:,ind_max])[ind_treat_2] # torch.max(dist_def_others_feet,1)[0].unsqueeze(1)
                # next_ballxy[ind_treat_2] = torch.diagonal(next_ballxys[ind_treat_2,ind_max])
                if self_theory2:
                    ball_Pball = Pballxy-ballxy_prev
                    dist_B_ballOF = torch.sqrt(torch.sum(ball_Pball**2,1))
                    time2catch_ballOF[ind_treat_2] = dist_B_ballOF[ind_treat_2]/pass_speed/self_max_v
                    if self_vel:
                        next_ballvel_[ind_treat_2] = (ball_Pball/dist_B_ballOF.unsqueeze(1).repeat(1,2)*pass_speed)[ind_treat_2]
                        next_ballxy_[ind_treat_2] = ballxy_prev[ind_treat_2] + next_ballvel_[ind_treat_2]/Fs*self_max_v
                    else:
                        next_ballxy_[ind_treat_2] = (ballxy_prev + ball_Pball/dist_B_ballOF.unsqueeze(1).repeat(1,2)*(pass_speed*self_max_v/Fs))[ind_treat_2]
                else:
                    next_ballxy_[ind_treat_2] = ballxy[ind_treat_2]

                posxy[:,10] = next_ballxy_
                if self_vel and self_theory2:
                    if dim==5:
                        next_ballvel_norm = torch.norm(next_ballvel_,dim=1).unsqueeze(1)
                        nonzeroindex = torch.where(next_ballvel_norm>1e-3)[0]
                        next_ballvel_ = next_ballvel_[nonzeroindex]/next_ballvel_norm[nonzeroindex].repeat(1,2)
                        vel_pred[nonzeroindex,10] = torch.cat([next_ballvel_,next_ballvel_norm[nonzeroindex]],1)
                    else:
                        vel_pred[:,10] = next_ballvel_

            if CF_flag and self_theory2: # defender
                dist_B_def_prev = torch.sqrt(torch.sum((Pballxy.unsqueeze(1).repeat(1,5,1)-posxy_prev[:,5:10,:])**2,2))
                dist_B_def_index_prev = torch.argsort(dist_B_def_prev)
                def_xy_prev = torch.diagonal(posxy_prev[:,5+dist_B_def_index_prev[:,0],:],dim1=0,dim2=1).transpose(1,0)
                def_Pball = Pballxy - def_xy_prev
                dist_def_Pball = torch.sqrt(torch.sum(def_Pball**2,1))
                if self_vel:
                    def_vel_unit = def_Pball/dist_def_Pball.unsqueeze(1).repeat(1,2)
                    # additional_vel = torch.exp(torch.clamp(x_pred.clone()[:,-1],max=1/2))/self_max_v
                    if dim==5:
                        vel_pred_norm_ = vel_pred_norm.clone()
                    else:
                        vel_pred_norm = torch.sqrt(torch.sum(vel_pred.clone()**2,2))+1e-4
                        # ball-ring distance
                        vec_B = Pballxy - L_Ring
                        Pballdist = torch.sqrt(torch.sum(vec_B**2,dim=1))+1e-4
                    
                    for b in range(batchSize):
                        ind_def = 5+dist_B_def_index_prev[b,0]
                        if Pballdist[b]<= 23*feet_m + 0.2286 and dist_def_Pball[b] > 6*0.3048: #  # 23ft 9in  (torch.sum(ind_treat_1==b)>0 or Ball_Hold[b]==0) and 
                            if dim==5:
                                posxy_def = posxy_prev[b,ind_def,:] + vel_pred_norm_[b,ind_def]/Fs*self_max_v
                            else:
                                vel_pred_norm__ = vel_pred_norm[b,ind_def] + defender_speed # additional_vel[b]
                                vel_pred_def = def_vel_unit[b]*vel_pred_norm__.unsqueeze(0).repeat(2)
                                
                                posxy_def = posxy_prev[b,ind_def,:] + vel_pred_def/Fs*self_max_v
                            dist_def_Pball2 = torch.sqrt(torch.sum((Pballxy[b] - posxy_def)**2))
                            if dist_def_Pball2 >= 4*0.3048:
                                if dim==5:
                                    vel_pred[b,ind_def] = torch.cat([def_vel_unit[b],vel_pred_norm_[b,ind_def].unsqueeze(0)],0)
                                else:
                                    vel_pred[b,ind_def] = vel_pred_def
                                posxy[b,ind_def,:] = posxy_prev[b,ind_def,:] + vel_pred_def/Fs*self_max_v
                                # posxy_def
                    
                else:
                    def_xy_new = def_xy_prev + def_Pball/dist_def_Pball.unsqueeze(1).repeat(1,2)*defender_speed/Fs*self_max_v
                    # (===pending===)
                    import pdb; pdb.set_trace()
                    for b in range(batchSize):
                        if (torch.sum(ind_treat_1==b)>0 or Ball_Hold[b]==0) and def_Pball[b] > 6*0.3048: # 
                            posxy[b,5+dist_B_def_index_prev[b,0],:] = def_xy_new[b,:]
            if self_vel:
                out[:,:self_x_dim_permuted] = torch.cat([posxy,vel_pred],2).reshape(-1,self_n_agents*dim)
                # out[:,:self_x_dim_permuted] = torch.cat([posxy.reshape(-1,self_n_agents*2),vel_pred.reshape(-1,self_n_agents*3)],1)
                # torch.cat([posxy,vel_pred],2).reshape(-1,self_n_agents*dim)
            else:
                out[:,:self_x_dim_permuted] = posxy.reshape(-1,self_n_agents*dim)  
            #if not CF_flag and i == 90:
            #    import pdb; pdb.set_trace()
            out[:,self_x_dim_permuted:-1] = torch.cat([torch.cat([shot_area,dist_def_feet.unsqueeze(1),dist_def_others_feet_max.unsqueeze(1)],1).to(torch.float),
                shot_prob2.unsqueeze(1),shot_prob_other_max,Clock.unsqueeze(1),ShotClock.unsqueeze(1),
                torch.cat([Ball_OF.unsqueeze(1),Ball_Hold.unsqueeze(1)],1).to(torch.float)],1) # ,torch.clamp(x_pred[:,-1:],max=1)


    elif not self_theory:
        out = x_pred

    else: 
        x_ = x.clone()
        if self_dataset == 'carla':
            Fs = 2
            out = x_
            dim = self_n_dim_each_permuted
            extra_dim = 4 
            # t-1
            loc_prev = x_prev[:,:-extra_dim].reshape(-1,self_n_agents,dim)[:,:,1:3]*self_max_p
            vel_norm_prev = x_prev[:,:-extra_dim].reshape(-1,self_n_agents,dim)[:,:,0]
            size_prev = x_prev[:,:-extra_dim].reshape(-1,self_n_agents,dim)[:,:,5:7]
            cossin_prev = x_prev[:,:-extra_dim].reshape(-1,self_n_agents,dim)[:,:,3:5]
            vel_prev = torch.cat([(vel_norm_prev*cossin_prev[:,:,0]).unsqueeze(2),(vel_norm_prev*cossin_prev[:,:,1]).unsqueeze(2)],2).reshape(-1,self_n_agents,2)
            # t-2
            vel_norm_prev2 = x_prev2[:,:-extra_dim].reshape(-1,self_n_agents,dim)[:,:,0]
            cossin_prev2 = x_prev2[:,:-extra_dim].reshape(-1,self_n_agents,dim)[:,:,3:5]
            vel_prev2 = torch.cat([(vel_norm_prev2*cossin_prev2[:,:,0]).unsqueeze(2),(vel_norm_prev2*cossin_prev2[:,:,1]).unsqueeze(2)],2).reshape(-1,self_n_agents,2)
            # t
            vel_norm_ = torch.clamp(x_pred.reshape(-1,self_n_agents,3)[:,:,0],max=2)
            vel_norm_ = torch.exp(vel_norm_)

            Constraint = True
            if Constraint:
                cosphi = torch.clamp(1-(x_pred.reshape(-1,self_n_agents,3)[:,:,1]*0.1)**2,min=0.9848, max=1) # 3.1415*10/180/Fs
                sinphi = torch.clamp(x_pred.reshape(-1,self_n_agents,3)[:,:,2]*0.1,min=-0.1736,max=0.1736)
                phi = torch.atan2(sinphi,cosphi)
                cosphi = torch.sqrt((1-sinphi**2))
            else:
                cossin_pred = torch.clamp(x_pred.reshape(-1,self_n_agents,3)[:,:,1:3],min=-1,max=1)
            # treatment                
            ind_treat_1 = torch.where(treatment==1)[0]
            additional_vel = torch.exp(torch.clamp(x_pred[:,-2],max=1))/10 #sigmoid torch.clamp(x_pred[:,-2],min=0,max=0.5)
            vel_norm = vel_norm_.clone()
            vel_norm[ind_treat_1,0] = vel_norm_[ind_treat_1,0] + additional_vel[ind_treat_1]
            
            # 
            if Constraint:
                cossin_pred = cossin_prev.clone()
                for ii in range(self_n_agents):
                    cossin_pred[:,ii,0] = cosphi[:,ii]*cossin_prev[:,ii,0] - sinphi[:,ii]*cossin_prev[:,ii,1]
                    cossin_pred[:,ii,1] = sinphi[:,ii]*cossin_prev[:,ii,0] + cosphi[:,ii]*cossin_prev[:,ii,1]
            vel_pred = cossin_pred*vel_norm.unsqueeze(2).repeat(1,1,2)

            # vel_pred = torch.cat([(vel_norm*cossin_pred[:,:,0]).unsqueeze(2),(vel_norm*cossin_pred[:,:,1]).unsqueeze(2)],2).reshape(-1,self_n_agents,2)
            # Integration using Simpson's rule
            loc = loc_prev + (vel_prev2+4*vel_prev+vel_pred)/6/Fs*self_max_v # vel_prev/Fs # 

            # true
            # cosphi = x[:,:-extra_dim].reshape(-1,self_n_agents,dim)[:,:,3]
            # sinphi = x[:,:-extra_dim].reshape(-1,self_n_agents,dim)[:,:,4]
            # loc_true = x[:,:-extra_dim].reshape(-1,self_n_agents,dim)[:,:,1:3]*self_max_p

            # agent*dim(7)+1(mileages)+1(collision)            
            out[:,:-extra_dim] = torch.cat([vel_norm.unsqueeze(2),loc/self_max_p,cossin_pred,size_prev],2).reshape(-1,self_n_agents*dim)
            #if torch.sum(ind_treat_1)>0: # torch.sum(torch.isinf(torch.abs(out))) or torch.sum(torch.isnan(out)):

            # mileages_true = x[:,self_x_dim_permuted]*self_max_y
            # mileages = x_prev[:,self_x_dim_permuted]*self_max_y + x_prev[:,0]/Fs*self_max_v
            mileages = x_prev[:,self_x_dim_permuted]*self_max_y + (x_prev2[:,0]+4*x_prev[:,0]+x_pred[:,0])/6/Fs*self_max_v
            out[:,-extra_dim] = mileages/self_max_y
            out[:,-3] = out[:,0]
            out[:,-2] = additional_vel
        elif 'boid' in self_dataset:
            out = x.clone()
            Fs = 1
            dim = 7
            loc_prev = x_prev[:,:-1].reshape(-1,self_n_agents,dim)[:,:,:2]*self_max_p
            cossin_prev = x_prev[:,:-1].reshape(-1,self_n_agents,dim)[:,:,2:4]
            vel_norm_prev = x_prev[:,:-1].reshape(-1,self_n_agents,dim)[:,:,4]
            vel_prev = x_prev[:,:-1].reshape(-1,self_n_agents,dim)[:,:,5:]

            center_prev = torch.mean(loc_prev,1).unsqueeze(1) # batch,agent,dim
            vec_ic_prev = loc_prev - center_prev.repeat(1,self_n_agents,1)   

            Diff = False
            if Diff:
                angvel_prev = torch.mean(torch.cross(F.pad(vec_ic_prev, (0, 1)),F.pad(vel_prev, (0, 1)),2)[:,:,-1],1)

            loc = loc_prev + vel_prev/Fs
            cosphi = torch.clamp(x_pred.reshape(-1,self_n_agents,2)[:,:,0],min=-1,max=1)
            sinphi = torch.clamp(x_pred.reshape(-1,self_n_agents,2)[:,:,1],min=-1,max=1)

            ind_treat_1 = torch.where(treatment==1)[0]

            # pull back on the center / intervention
            loc_norm = torch.sqrt(torch.sum((loc_prev-torch.mean(loc_prev,1).unsqueeze(1).repeat(1,self_n_agents,1))**2,2)+1e-4)
            # if torch.sum(torch.sum(loc_norm>6))>0:
           
            cosbeta = torch.cos(torch.ones(batchSize,)*3.1415*30/180/Fs)
            sinbeta = torch.sin(torch.ones(batchSize,)*3.1415*30/180/Fs)
            tmp_cross_all = torch.cross(F.pad(loc_prev, (0, 1)),F.pad(vel_prev, (0, 1)))[:,:,-1]

            for ii in range(self_n_agents):
                tmp_cross = tmp_cross_all[:,ii] # torch.cross(F.pad(loc_prev[:,ii], (0, 1)),F.pad(vel_prev[:,ii], (0, 1)))[:,-1]
                # intervention
                other_loc = torch.cat([loc_prev[:,:i],loc_prev[:,i+1:]],1)
                dist_i = torch.sqrt(torch.sum((other_loc-loc_prev[:,i].unsqueeze(1).repeat(1,self_n_agents-1,1))**2,2)+1e-4)
                other_vel = torch.cat([vel_prev[:,:i],vel_prev[:,i+1:]],1)

                if torch.sum(loc_norm[:,ii]>6)>0: # 6
                    cross_positive = torch.where((tmp_cross>=0)&(loc_norm[:,ii]>6))[0]
                    cross_negative = torch.where((tmp_cross<0)&(loc_norm[:,ii]>6))[0]
                    cosphi[cross_positive,ii] = cosbeta[cross_positive] # -tmp_cossin[cross_positive,1]
                    sinphi[cross_positive,ii] = sinbeta[cross_positive] #  tmp_cossin[cross_positive,0]
                    cosphi[cross_negative,ii] = cosbeta[cross_negative] # tmp_cossin[cross_negative,1]
                    sinphi[cross_negative,ii] = -sinbeta[cross_negative] # tmp_cossin[cross_negative,0] 

                # intervention       
                if len(torch.where((loc_norm[:,ii]>1)&(loc_norm[:,ii]<=4)&(treatment==1))[0])>0:
                    for bb in range(batchSize):
                        within_align = torch.where((dist_i[bb]<=4)&(dist_i[bb]>1))
                        # other_vel_mean = torch.mean(other_vel[bb,within_align[0]],0) if within_align[0].shape[0]>1 else other_vel[bb,within_align[0]]
                        # if torch.mean(tmp_cross_all[bb,within_align[0]])>=0
                        if torch.mean(tmp_cross_all[bb,within_align[0]])>=0 and torch.min(dist_i[bb])>0.5:
                            cosphi[bb,ii] = cosbeta[0]
                            sinphi[bb,ii] = sinbeta[0]
                        elif torch.mean(tmp_cross_all[bb,within_align[0]])<0 and torch.min(dist_i[bb])>0.5:
                            cosphi[bb,ii] = cosbeta[0]
                            sinphi[bb,ii] = -sinbeta[0]
         
            cosphi = torch.clamp(cosphi, min=cosbeta[0], max=1) 
            sinphi = torch.clamp(sinphi, min=-sinbeta[0], max=sinbeta[0]) 
            phi = torch.atan2(sinphi,cosphi)

            vel = vel_prev.clone()
            for ii in range(self_n_agents):
                vel[:,ii,0] = torch.cos(phi[:,ii])*vel_prev[:,ii,0] - torch.sin(phi[:,ii])*vel_prev[:,ii,1]
                vel[:,ii,1] = torch.sin(phi[:,ii])*vel_prev[:,ii,0] + torch.cos(phi[:,ii])*vel_prev[:,ii,1]

            center = torch.mean(loc,1).unsqueeze(1) # batch,agent,dim
            vec_ic = loc - center.repeat(1,self_n_agents,1)
            angvel = torch.abs(torch.mean(torch.cross(F.pad(vec_ic, (0, 1)),F.pad(vel, (0, 1)),2)[:,:,-1],1))
            out[:,:-1] = torch.cat([loc/self_max_p,cosphi.unsqueeze(2),sinphi.unsqueeze(2),vel_norm_prev.unsqueeze(2),vel],2).reshape(-1,self_n_agents*dim)
            out[:,-1] = angvel if not Diff else angvel-angvel_prev
            if torch.abs(out[0,0])>20:
                import pdb; pdb.set_trace()
        
    #else: # 'synthetic' in self_dataset:

    return out