import argparse, random
import numpy as np
import glob
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from tqdm import tqdm
import multiprocessing as mp

import os, pickle
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import data_loader_nba
import data_loader_carla
import data_loader_boid
from model_GDSW import GDSW
from model_GVCRN import GVCRN
from model_RNN import RNN
from utils import batch_error, compute_x_ind, std_ste

from torch.multiprocessing import Pool, Process, set_start_method

HIDDEN_SIZE = 32
CUDA = True
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["HDF5_USE_FILE_LOCKING"]="FALSE"


def compute_loss(a_or_ipw_outputs,f_outcome_out,x_out,fr_targets,targets,x_inputs_,L_kl,lengths,Pred_X,criterion,args,obs_w,batchSize,device,inference=False):
    a_or_ipw_loss = torch.zeros(1).to(device)
    outcome_loss = torch.zeros(1).to(device)
    fail_loss = torch.zeros(1).to(device)
    curve_loss = torch.zeros(1).to(device)
    L_recX = torch.zeros(1).to(device)
    time_length = x_inputs_.shape[1]-1
    if not args.variable_length:
        non_nan = None
        mean_time_length = time_length-args.burn_in0-1
    else:
        mean_time_length = torch.mean(lengths-1)-args.burn_in0-1
    t_pred = 1 if args.y_pred else 0

    if "B" in args.model or "RNN" in args.model: # GradientReversal or negative gradient
        if "B" in args.model:
            a_or_ipw_outputs = torch.cat(a_or_ipw_outputs, dim=1)
            ps = torch.sigmoid(a_or_ipw_outputs)
            for i in range(args.burn_in0+1,time_length): # time
                # propensity score
                if args.variable_length:
                    non_nan = lengths>=obs_w-i
                    if "Ne" in args.model:
                        a_or_ipw_loss -= criterion(non_nan*ps[:,i], non_nan*fr_targets[:, i].float(), reduction='sum')
                    else:
                        a_or_ipw_loss += criterion(non_nan*ps[:,i], non_nan*fr_targets[:, i].float(), reduction='sum')
                else:
                    if "Ne" in args.model:
                        a_or_ipw_loss -= criterion(ps[:,i], fr_targets[:, i].float(), reduction='sum')
                    else:
                        a_or_ipw_loss += criterion(ps[:,i], fr_targets[:, i].float(), reduction='sum')
            a_or_ipw_loss = a_or_ipw_loss/mean_time_length/batchSize
        # y,x,fail
        x_dim = int(args.x_dim_permuted//args.n_agents)
        n_agents = args.n_agents
        n_agents_ = n_agents if "T" in args.model and args.dim_rec_global == 0 else n_agents+1
        
        for i in range(args.burn_in0+1,time_length): # time
            if args.variable_length:
                non_nan = lengths>=obs_w-i
                outcome_loss += torch.sum(non_nan.squeeze()*torch.abs(f_outcome_out[:,i] - targets[:,i+t_pred]) )
                if Pred_X:
                    for k in range(n_agents_): # n_agents+args.dim_rec_global):
                        if k < n_agents:
                            if 'carla' in args.data:
                                L_recX += batch_error(x_out[:,i,k*x_dim:k*x_dim+3], x_inputs_[:,i,k*x_dim:k*x_dim+3],index=non_nan)
                            else: # nba
                                L_recX += batch_error(x_out[:,i,k*x_dim:(k+1)*x_dim], x_inputs_[:,i,k*x_dim:(k+1)*x_dim],index=non_nan)
                        else:
                            k2 = args.x_dim_permuted+args.dim_rec_global 
                            L_recX += batch_error(x_out[:,i,args.x_dim_permuted:k2+1], x_inputs_[:,i,args.x_dim_permuted:k2+1],index=non_nan)

                if 'carla' in args.data and Pred_X:
                    fail_loss += torch.sum(non_nan.squeeze()*torch.abs(x_out[:,i,-1] - x_inputs_[:,i+1,-1]) ) 
                    for k in range(n_agents_):
                        curve_loss += batch_error(x_out[:,i,k*x_dim+3:k*x_dim+5], x_out[:,i-1,k*x_dim+3:k*x_dim+5],index=non_nan)

            else:
                outcome_loss += torch.sum(torch.abs(f_outcome_out[:,i] - targets[:,i+t_pred]) )
                if Pred_X:
                    for k in range(n_agents+args.dim_rec_global):
                        if k < n_agents:
                            if 'boid' in args.data:
                                try: L_recX += batch_error(x_out[:,i,k*x_dim+2:k*x_dim+4], x_inputs_[:,i,k*x_dim+2:k*x_dim+4])
                                except: import pdb; pdb.set_trace()
                        else:
                            k2 = args.x_dim_permuted+k-n_agents
                            L_recX += batch_error(x_out[:,i,k2:k2+1], x_inputs_[:,i,k2:k2+1])

        L_recX /= mean_time_length*(n_agents+args.dim_rec_global)*batchSize # 
        fail_loss /= mean_time_length*(n_agents-1)*batchSize
        curve_loss /= mean_time_length*(n_agents-1)*batchSize
    else:
        a_or_ipw_outputs = torch.cat(a_or_ipw_outputs, dim=1)
        ps = torch.sigmoid(a_or_ipw_outputs)
        weights = torch.zeros(a_or_ipw_outputs.shape)
        for i in range(args.burn_in0+1,time_length):
            a_or_ipw_pred_norm = a_or_ipw_outputs[:,i]
            
            if args.variable_length:
                non_nan = lengths>=obs_w-i
                a_or_ipw_loss += criterion(non_nan*a_or_ipw_pred_norm, non_nan*fr_targets[:, i].float(), reduction='sum')
            else:
                a_or_ipw_loss += criterion(a_or_ipw_pred_norm, fr_targets[:, i].float(), reduction='sum')
                
            for j in range(a_or_ipw_outputs.size(0)): # batch
                    p_treated = torch.where(fr_targets[j] == 1)[0].size(0) / fr_targets.size(1)
                    if fr_targets[j,i] != 0:
                        weights[j,i] += p_treated / ps[j,i] if ps[j,i] > 1e-3 else p_treated / 1e-3
                    else:
                        weights[j,i] += (1 - p_treated) / (1 - ps[j,i]) if 1-ps[j,i] > 1e-3 else (1 - p_treated) / 1e-3

            weights2 = torch.where(weights[:,i] >= 100, torch.Tensor([100]), weights[:,i])
            weights3 = torch.where(weights2 <= 0.01, torch.Tensor([0.01]), weights2)
            if args.variable_length:
                outcome_loss += torch.sum(weights3*torch.abs(non_nan*f_outcome_out[:,i] - non_nan*targets[:,i+t_pred]) )
            else:
                outcome_loss += torch.sum(weights3*torch.abs(f_outcome_out[:,i] - targets[:,i+t_pred]) )

        a_or_ipw_loss = a_or_ipw_loss/mean_time_length/batchSize
    outcome_loss = outcome_loss/mean_time_length/batchSize
    L_kl /= batchSize
    if torch.sum(torch.isnan(outcome_loss))>0:
        import pdb; pdb.set_trace()
    return a_or_ipw_loss, outcome_loss, fail_loss, curve_loss, L_recX, L_kl

def display_loss(a_or_ipw_losses, outcome_losses, L_recXs, L_fails, L_curves, L_kls, epoch, args, Pred_X):

    epoch_losses_a_or_ipw = np.mean(a_or_ipw_losses)
    outcome_losses = np.mean(outcome_losses)
    if Pred_X and 'carla' in args.data:
        str_fail_loss = ', L_fail: {:.4f}'.format(np.mean(L_fails))
        str_curve_loss = ', L_curve: {:.4f}'.format(np.mean(L_curves))
    else: 
        str_fail_loss = ''
        str_curve_loss = ''
    if not 'RNN' in args.model:
        print('Epoch: {}, a_or_ipw loss: {:.4f}, Outcome loss: {:.4f}'.format(epoch, epoch_losses_a_or_ipw, outcome_losses), flush=True)
        if Pred_X:
            L_recXs = np.mean(np.sqrt(L_recXs))
            if 'V' in args.model:
                L_kls = np.mean(L_kls)
                print('Epoch: {}, L_kls: {:.4f}, L_recXs: {:.4f}'.format(epoch, L_kls, L_recXs)+str_fail_loss+str_curve_loss, flush=True)
            else:
                print('Epoch: {}, L_recXs: {:.4f}'.format(epoch, L_recXs)+str_fail_loss+str_curve_loss, flush=True)
    else:
        L_recXs = np.mean(L_recXs)
        print('Epoch: {}, Outcome train loss: {:.4f}, L_recXs: {:.4f}'.format(epoch, outcome_losses, L_recXs)+str_fail_loss+str_curve_loss, flush=True)

def trainInitIPTW(train_loader, val_loader,test_loader, model, args, epochs, optimizer, criterion, 
                  use_cuda=False, save_model=None,TEST=False):

    if use_cuda:
        print("====> Using CUDA device: ", torch.cuda.current_device(), flush=True)
        model.cuda()
        model = model.to('cuda')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    burn_in = args.burn_in
    obs_w = args.observation_window
    # Train network
    best_loss_val = torch.tensor(float('inf')).to(device)
    if not TEST and not args.has_GT:
        CF = False
    else:
        CF = True
    Pred_X = True if 'V' in args.model or 'RNN' in args.model or 'X' in args.model else False

    if args.cont:
        print('args.cont = True')
        if os.path.exists(save_model): 
            model = torch.load(save_model)      
            print('best model was loaded')   
        else:
            print('args.cont = True but file did not exist')
    no_update = 0 
    if not TEST:
        for epoch in range(epochs):
            a_or_ipw_losses = []
            outcome_losses = []
            L_kls = []
            L_recXs = []
            L_fails = []
            L_curves = []

            for x_inputs, x_static_inputs, x_fr_inputs, targets, lengths, _,_,_,_,_ in tqdm(train_loader): # train_loader # tqdm(train_loader): # x_all, x_fr_all, targets_all,x_fr_opt 
                model.train()
                # train
                optimizer.zero_grad()

                fr_targets = x_fr_inputs
                if use_cuda:
                    x_inputs, x_static_inputs, x_fr_inputs = x_inputs.cuda(), x_static_inputs.cuda(), x_fr_inputs.cuda()
                    targets, fr_targets = targets.cuda(), fr_targets.cuda()
                    if args.variable_length:
                        lengths = lengths.cuda()

                if args.data == 'nba':# or args.data == 'carla':
                    x_inputs_ = x_inputs[:,:,:-1]
                else:
                    x_inputs_ = x_inputs
                x_inputs__ = x_inputs_
                
                if not 'V' in args.model and not 'RNN' in args.model:
                    a_or_ipw_outputs, f_outcome_out, _, _, x_out,_ = model(x_inputs_, x_static_inputs, fr_targets, targets, cf_treatment=None, lengths=lengths)
                    L_kl = torch.zeros(1).to(device)
                else:
                    a_or_ipw_outputs, f_outcome_out, _, _, L_kl, x_out,_ = model(x_inputs_, x_static_inputs, fr_targets, targets, cf_treatment=None, Train=True, lengths=lengths)
                f_treatment = torch.where(fr_targets.sum(1) > 0, torch.Tensor([1]), torch.Tensor([0]))

                f_outcome_out = torch.stack(f_outcome_out,dim=1).squeeze()
                batchSize = f_outcome_out.shape[0]

                a_or_ipw_loss, outcome_loss, fail_loss, curve_loss, L_recX, L_kl = compute_loss(a_or_ipw_outputs,f_outcome_out,x_out,fr_targets,targets,x_inputs__,L_kl,lengths,Pred_X,criterion,args,obs_w,batchSize,device)
                if not 'RNN' in args.model:
                    loss = a_or_ipw_loss * args.lambda_weight + outcome_loss
                else:
                    loss = outcome_loss
                if Pred_X:
                    loss += L_recX[0]*args.lambda_X
                    if 'V' in args.model:
                        loss += L_kl[0]*args.lambda_KL
                if 'carla' in args.data:
                    loss += fail_loss*args.lambda_event
                    loss += curve_loss*args.lambda_event

                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 20)

                optimizer.step()

                a_or_ipw_losses.append(a_or_ipw_loss.item())
                outcome_losses.append(outcome_loss.item())
                if Pred_X:
                    L_recXs.append(L_recX.item())
                    if 'V' in args.model:
                        L_kls.append(L_kl.item())
                if 'carla' in args.data:
                    L_fails.append(fail_loss.item()) 
                    L_curves.append(curve_loss.item())   
            
            display_loss(a_or_ipw_losses, outcome_losses, L_recXs, L_fails, L_curves, L_kls, epoch, args, Pred_X)    

            # validation
            print('Validation:')
            pehe_val, _, mse_val, loss_val = model_eval(model, val_loader, criterion, args, epoch, eval_use_cuda=use_cuda, CF=CF)

            if loss_val < best_loss_val:
                best_loss_val = loss_val

                if save_model:
                    print('Best model. Saving...\n')
                    torch.save(model, save_model)
            elif np.isnan(loss_val) or (epoch==0 and loss_val == best_loss_val):
                print('loss is nan or inf')
                import pdb; pdb.set_trace()
            else:
                no_update += 1
                if no_update >= 3:
                    try: model = torch.load(save_model)    
                    except: import pdb; pdb.set_trace()  
                    print('since no update continues, best model was loaded')   
                    no_update = 0
    else:
        epoch = 0
                
    print('Test:')
    model = torch.load(save_model)
    rmse_y_CF_max,rmse_best_timing,rmse,_ = model_eval(model, test_loader,criterion, args, epoch, eval_use_cuda=use_cuda, save=True, TEST=True)

def detach(data,eval_use_cuda):
    if eval_use_cuda:
        return data.to('cpu').detach().data.numpy()
    else:
        return data.detach().data.numpy()

def transfer_data(model, dataloader, criterion, args, epoch, eval_use_cuda=False, save=False, TEST=False):
    burn_in = args.burn_in
    burn_in_test = args.burn_in_test
    obs_w = args.observation_window
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('TEST ='+str(TEST))
    Pred_X = True if 'V' in args.model or 'RNN' in args.model or 'X' in args.model else False

    with torch.no_grad():
        model.eval()
        a_or_ipw_losses = []
        outcome_losses = []
        L_kls = []
        L_recXs = []
        L_fails = []
        L_curves = []

        f_outcome_outputs = []
        cf_outcome_outputs = []
        f_outcome_outputs_long = []
        f_outcome_true = []
        cf_outcome_true = []
        treatment_f = []
        treatment_all = []
        treatment_opts = []
        a_or_ipw_true = []
        loss_all = []
        x_outs = []
        x_outs_cf = []
        x_outs_true = []
        x_inputs_f, x_static_inputs_f, lengths_all,indices_all = [],[],[],[]

        for x_inputs, x_static_inputs, x_fr_inputs, targets, lengths, indices, x_fr_all, x_all, targets_all, x_fr_opt in dataloader:
            fr_targets = x_fr_inputs
            if eval_use_cuda:
                x_inputs, x_static_inputs, x_fr_inputs = x_inputs.cuda(), x_static_inputs.cuda(), x_fr_inputs.cuda()
                x_fr_all, targets, fr_targets = x_fr_all.cuda(), targets.cuda(), fr_targets.cuda()
                if args.variable_length:
                    lengths = lengths.cuda()
                if args.has_GT:
                    x_all, targets_all = x_all.cuda(), targets_all.cuda()

            if args.data == 'nba':# or args.data == 'carla':
                x_inputs_ = x_inputs[:,:,:-1].clone()
                x_all = x_all[:,:,:-1]
            else:
                x_inputs_ = x_inputs.clone()

            x_inputs__ = x_inputs_.clone()

            if not TEST and not args.has_GT:
                x_fr_all_ = None
                CF = False
            else:
                x_fr_all_ = x_fr_all
                CF = True
           

            # counterfactual treatments
            if not 'V' in args.model and not 'RNN' in args.model:
                a_or_ipw_outputs, f_outcome_out, cf_outcome_out, _, x_out, x_out_cf = model(x_inputs_, x_static_inputs, fr_targets, targets, cf_treatment=x_fr_all_, burn_in=burn_in, lengths=lengths)
                if not args.has_GT: # nba
                    a_or_ipw_outputs, f_outcome_out2, _, _, x_out, _ = model(x_inputs_, x_static_inputs, fr_targets, targets, cf_treatment=None, burn_in=burn_in_test, lengths=lengths)
                L_kl = torch.zeros(1).to(device)
            else:
                a_or_ipw_outputs, f_outcome_out, cf_outcome_out, _, L_kl, x_out, x_out_cf = model(x_inputs_, x_static_inputs, fr_targets, targets, cf_treatment=x_fr_all_, burn_in=burn_in, lengths=lengths)
                if not args.has_GT: # nba
                    a_or_ipw_outputs, f_outcome_out2, _, _, L_kl, x_out, _ = model(x_inputs_, x_static_inputs, fr_targets, targets, cf_treatment=None, burn_in=burn_in_test, lengths=lengths)
            
            if CF:
                cf_outcome_out = cf_outcome_out.permute(2,1,0,3)
            f_outcome_out = torch.stack(f_outcome_out,dim=1).squeeze()
            if not args.has_GT: # nba
                f_outcome_out_long = f_outcome_out.clone()
                f_outcome_out = torch.stack(f_outcome_out2,dim=1).squeeze()
            # x_out = torch.stack(x_out,dim=1)
            batchSize = f_outcome_out.shape[0]

            a_or_ipw_loss, outcome_loss, fail_loss, curve_loss, L_recX, L_kl = compute_loss(a_or_ipw_outputs,f_outcome_out,x_out,fr_targets,targets,x_inputs__,L_kl,lengths,Pred_X,criterion,args,obs_w,batchSize,device,inference=True)
            a_or_ipw_losses.append(a_or_ipw_loss.item())
            outcome_losses.append(outcome_loss.item())
            if Pred_X:
                L_recXs.append(L_recX.item())
                if 'V' in args.model:
                    L_kls.append(L_kl.item())
            if 'carla' in args.data:
                L_fails.append(fail_loss.item())   
                L_curves.append(curve_loss.item())  

            if not 'RNN' in args.model:
                loss = a_or_ipw_loss * args.lambda_weight + outcome_loss
            else:
                loss = outcome_loss
            if Pred_X:
                loss += L_recX[0]*args.lambda_X
                if 'V' in args.model:
                    loss += L_kl[0]*args.lambda_KL
            if 'carla' in args.data:
                loss += fail_loss*args.lambda_event
                loss += curve_loss*args.lambda_event

            # detach
            for i in range(len(a_or_ipw_outputs)):
                try: a_or_ipw_outputs[i] = detach(a_or_ipw_outputs[i],eval_use_cuda)
                except: import pdb; pdb.set_trace()
            fr_targets = detach(fr_targets,eval_use_cuda)
            targets = detach(targets,eval_use_cuda)
            f_outcome_out = detach(f_outcome_out,eval_use_cuda)
            if CF:
                cf_outcome_out = detach(cf_outcome_out,eval_use_cuda)
            if not args.has_GT: # nba
                f_outcome_out_long = detach(f_outcome_out_long,eval_use_cuda)
            loss = detach(loss,eval_use_cuda)

            if args.has_GT:
                targets_all = detach(targets_all,eval_use_cuda)
                x_all = detach(x_all,eval_use_cuda)
            x_fr_all = detach(x_fr_all,eval_use_cuda)
            if Pred_X:
                x_out = detach(x_out,eval_use_cuda)
                x_out_cf = detach(x_out_cf,eval_use_cuda)

            x_inputs = detach(x_inputs,eval_use_cuda)
            x_static_inputs = detach(x_static_inputs,eval_use_cuda)
            # x_fr_inputs = detach(x_fr_inputs,eval_use_cuda)
            if args.variable_length:
                lengths = detach(lengths,eval_use_cuda)
                if save:
                    indices = detach(indices,eval_use_cuda)

            # append            
            treatment_f.append(fr_targets) 
            treatment_all.append(x_fr_all)
            if args.has_GT:
                x_fr_opt = x_fr_opt.detach().data.numpy()   
                treatment_opts.append(x_fr_opt)

            a_or_ipw_true.append(np.where(fr_targets.sum(1) > 0, 1, 0))
            f_outcome_true.append(targets)
            if args.has_GT:
                cf_outcome_true.append(targets_all)
            f_outcome_outputs.append(f_outcome_out)
            if CF:
                cf_outcome_outputs.append(cf_outcome_out)
            if not args.has_GT:
                f_outcome_outputs_long.append(f_outcome_out_long)
            loss_all.append(loss)
            if Pred_X:
                x_outs.append(x_out)
                x_outs_cf.append(x_out_cf)
                x_outs_true.append(x_all)

            x_inputs_f.append(x_inputs)
            x_static_inputs_f.append(x_static_inputs)
            # x_fr_inputs_f.append(x_fr_inputs)
            if args.variable_length:
                lengths_all.append(lengths)
                indices_all.append(indices)
            
        display_loss(a_or_ipw_losses, outcome_losses, L_recXs, L_fails, L_curves, L_kls, epoch, args, Pred_X)
        # concatenate
        a_or_ipw_true = np.concatenate(a_or_ipw_true).transpose()
        f_outcome_true = np.concatenate(f_outcome_true)
        if args.has_GT:
            cf_outcome_true = np.concatenate(cf_outcome_true)
            treatment_opts = np.concatenate(treatment_opts)
        else:
            cf_outcome_true = None
            treatment_opts = None
        f_outcome_outputs = np.concatenate(f_outcome_outputs)
        if CF:
            cf_outcome_outputs = np.concatenate(cf_outcome_outputs)
        if not args.has_GT:
            f_outcome_outputs_long = np.concatenate(f_outcome_outputs_long)
        # loss_all = np.concatenate(loss_all)
        loss_all = np.mean(loss_all)
        if Pred_X:
            x_outs = np.concatenate(x_outs,0)
            x_outs_cf = np.concatenate(x_outs_cf,1)
            x_outs_true = np.concatenate(x_outs_true)
        else:
            x_outs = None
            x_outs_true = None 

        x_inputs_f = np.concatenate(x_inputs_f)
        x_static_inputs_f = np.concatenate(x_static_inputs_f)
        if args.data == 'nba' or args.data == 'carla':
            lengths_all = np.concatenate(lengths_all)
            if save: 
                indices_all = np.concatenate(indices_all)
        else:
            lengths_all,indices_all = [],[]
        
        # for saving
        if args.has_GT:
            outcomes = [f_outcome_true, cf_outcome_true, f_outcome_outputs, cf_outcome_outputs]
        else:
            outcomes = [f_outcome_true, cf_outcome_true, f_outcome_outputs, cf_outcome_outputs, f_outcome_outputs_long]
        covariates = [x_inputs_f, x_outs, x_outs_true, x_static_inputs_f, x_outs_cf]

        if save: 
            others = [treatment_f, treatment_all, treatment_opts, lengths_all, indices_all, loss_all]
        else:
            others = [treatment_f, treatment_all, treatment_opts, lengths_all, loss_all]
        return outcomes, covariates, others

def model_eval(model, dataloader, criterion, args, epoch, eval_use_cuda=False, save=False, TEST=False, CF=True):
    burn_in = args.burn_in #+ 1
    burn_in_test = args.burn_in_test #+ 1
    burn_in_ = burn_in if args.has_GT else burn_in_test
    Pred_X = True if 'V' in args.model or 'RNN' in args.model or 'X' in args.model else False

    outcomes, covariates, others = transfer_data(model, dataloader, criterion, args, epoch, eval_use_cuda, save=save, TEST=TEST)
    if args.has_GT:
        f_outcome_true, cf_outcome_true, f_outcome_outputs, cf_outcome_outputs = outcomes
    else:
        f_outcome_true, cf_outcome_true, f_outcome_outputs, cf_outcome_outputs, _ = outcomes
    x_inputs_f, x_outs, x_outs_true, x_static_inputs_f, x_outs_cf = covariates
    if save: 
        treatment_f, treatment_all, treatment_opts, lengths_all, indices_all, loss_all = others
    else:
        treatment_f, treatment_all, treatment_opts, lengths_all, loss_all = others
   
    if args.data == 'nba':
        n_agents = args.n_agents - 1
        x_dim_permuted_ = n_agents*2 if not args.vel else n_agents*4 #  4 args.x_dim_permuted-2
    else:
        n_agents = args.n_agents
        x_dim_permuted_ = args.x_dim_permuted
    n_dim_each_permuted = int(x_dim_permuted_//n_agents)

    model = args.model
    std = False 
    t_pred = 1 if args.y_pred else 0
    if args.has_GT:
        # ouctome
        if args.y_pred:
            y_all_true = cf_outcome_true[:,burn_in_+1:].reshape((cf_outcome_true.shape[0],-1))
        else:
            y_all_true = cf_outcome_true[:,burn_in_:-1].reshape((cf_outcome_true.shape[0],-1))
        y_pred_true = cf_outcome_outputs[:,burn_in_:,:,0].reshape((cf_outcome_true.shape[0],-1))
        rmse = mean_squared_error(y_all_true,y_pred_true, multioutput='raw_values',squared=False)
        # best timing
        if 'boid' in args.data:
            if args.y_pred:
                cfo_true_last = cf_outcome_true[:,burn_in_+1:,:]
                tau_true = cf_outcome_true[:,burn_in_+1:,:5] - np.repeat(cf_outcome_true[:,burn_in_+1:,-1,np.newaxis],5,axis=2)
            else:
                cfo_true_last = cf_outcome_true[:,burn_in_:-1,:]
                tau_true = cf_outcome_true[:,burn_in_:-1,:5] - np.repeat(cf_outcome_true[:,burn_in_:-1,-1,np.newaxis],5,axis=2)
            cfo_last = cf_outcome_outputs[:,burn_in_:,:,0]
            cfo_true_last = np.max(np.abs(cfo_true_last),1)
            cfo_last = np.max(np.abs(cfo_last),1)
            cfo_last_max_ind = np.argmax(cfo_last,1)
            best_timing_true = np.argmax(cfo_true_last,1)
            timing_or_diff = np.sqrt((best_timing_true-cfo_last_max_ind)**2)
            tau_pred = cf_outcome_outputs[:,burn_in_:,:5,0] - np.repeat(cf_outcome_outputs[:,burn_in_:,-1],5,axis=2)

        if args.data == 'carla':
            diff_f = cf_outcome_true[:,-2+t_pred,0] - cf_outcome_true[:,-2+t_pred,1]
            diff_cf = cf_outcome_outputs[:,-1,0,0] - cf_outcome_outputs[:,-1,1,0]
            timing_or_diff = np.abs(diff_f-diff_cf).reshape((cf_outcome_true.shape[0],-1))
            if args.y_pred:
                tau_true = cf_outcome_true[:,burn_in_+1:,0:1] - cf_outcome_true[:,burn_in_+1:,1:]
            else:
                tau_true = cf_outcome_true[:,burn_in_:-1,0:1] - cf_outcome_true[:,burn_in_:-1,1:]
            tau_pred = cf_outcome_outputs[:,burn_in_:,0] - cf_outcome_outputs[:,burn_in_:,1]

        # PEHE
        N = tau_true.shape[0]
        PEHE = np.sqrt(np.sum((tau_true-tau_pred)**2,axis=0)/N).reshape(-1,)
        # ATE
        ATE = np.abs(np.sum(tau_true,axis=0)/N - np.sum(tau_pred,axis=0)/N).reshape(-1,)
        if args.data == 'carla':
            timing_or_diff = PEHE
            
        if not Pred_X:
            # rmse_max,rmse_best_timing_,rmse_ = result
            #if 'boid' in args.data:
            #    timing_or_diff = rmse_best_timing
            print(model+': '
            +' ' + '{:.3f}'.format(np.mean(rmse))+' $\pm$ '+'{:.3f}'.format(std_ste(rmse,std=std))+' &'
            +' ' + '{:.3f}'.format(np.mean(PEHE))+' $\pm$ '+'{:.3f}'.format(std_ste(PEHE,std=std))+' &'
            +' ' + '{:.3f}'.format(np.mean(timing_or_diff))+' $\pm$ '+'{:.3f}'.format(std_ste(timing_or_diff,std=std))+' & ---'
            )
            rmse_cf_x = None
            if args.data == 'carla':
                loss_all = np.mean(rmse) + np.mean(timing_or_diff) # 1/10
            elif 'boid' in args.data:
                loss_all = np.mean(rmse) + np.mean(PEHE) + np.mean(timing_or_diff) # 2/1
        else:
            # rmse_max,rmse_best_timing_,rmse_,rmse_cf_x_ = result
            #if 'boid' in args.data:
            #    timing_or_diff = rmse_best_timing

            n_dim_x = 2
            n_agents = args.n_agents
            if args.data == 'carla':
                n_dim_agent = 7
                ind_x = np.concatenate([np.arange(1,n_agents*n_dim_agent,n_dim_agent)[:,np.newaxis],np.arange(2,n_agents*n_dim_agent,n_dim_agent)[:,np.newaxis]],1)
            elif 'boid' in args.data:
                n_dim_agent = 7
                ind_x = np.concatenate([np.arange(0,n_agents*n_dim_agent,n_dim_agent)[:,np.newaxis],np.arange(1,n_agents*n_dim_agent,n_dim_agent)[:,np.newaxis]],1)
            ind_x = ind_x.reshape((-1))

            x_outs_true_trim = x_outs_true[:,burn_in:,ind_x].transpose((0,1,3,2)).reshape((-1,n_agents,n_dim_x)).reshape((-1,n_dim_x))
            
            x_outs_trim = x_outs_cf[:,:,burn_in:,ind_x].transpose((1,2,0,3)).reshape((-1,n_agents,n_dim_x)).reshape((-1,n_dim_x))
            #x_outs_true_trim = x_outs_true[:,burn_in:,x_dim_permuted_].transpose((0,1,3,2)).reshape((-1,n_agents,n_dim_each_permuted)).reshape((-1,n_dim_each_permuted))
            #x_outs_trim = x_outs[:,:,burn_in:,:x_dim_permuted_].transpose((1,2,0,3)).reshape((-1,n_agents,n_dim_each_permuted)).reshape((-1,n_dim_each_permuted))
            rmse_cf_x = mean_squared_error(x_outs_true_trim.T,x_outs_trim.T, multioutput='raw_values',squared=False)
            print(model+': '
            +' ' + '{:.3f}'.format(np.mean(rmse))+' $\pm$ '+'{:.3f}'.format(std_ste(rmse,std=std))+' & '
            +' ' + '{:.3f}'.format(np.mean(PEHE))+' $\pm$ '+'{:.3f}'.format(std_ste(PEHE,std=std))+' & '
            +' ' + '{:.3f}'.format(np.mean(timing_or_diff))+' $\pm$ '+'{:.3f}'.format(std_ste(timing_or_diff,std=std))+' &'
            +' ' + '{:.3f}'.format(np.mean(rmse_cf_x))+' $\pm$ '+'{:.4f}'.format(std_ste(rmse_cf_x,std=std))+' '
            )
            if args.data == 'carla':
                loss_all = np.mean(rmse) + np.mean(timing_or_diff) + 10*np.mean(rmse_cf_x) # 1/10/10
            elif 'boid' in args.data:
                loss_all = np.mean(rmse) + np.mean(PEHE) + np.mean(timing_or_diff) + np.mean(rmse_cf_x) # 2/1/1/10
        rmse_max = None
        rmse_best_timing = timing_or_diff
        result = [rmse_best_timing,rmse,rmse_cf_x]

    else:
        if args.y_pred:
            rmse = mean_squared_error(f_outcome_true[:,burn_in_+1:].T,f_outcome_outputs[:,burn_in_:].T, multioutput='raw_values',squared=False)
        else:
            rmse = mean_squared_error(f_outcome_true[:,burn_in_:-1].T,f_outcome_outputs[:,burn_in_:].T, multioutput='raw_values',squared=False)
        if Pred_X:
            if args.vel:
                try: x_outs_true_trim = (x_outs_true[:,burn_in_:,:x_dim_permuted_].reshape((-1,n_agents,n_dim_each_permuted)))[:,:,:2].reshape((-1,2))
                except: import pdb; pdb.set_trace()
                x_outs_trim = x_outs[:,burn_in_:,:x_dim_permuted_].reshape((-1,n_agents,n_dim_each_permuted))[:,:,:2].reshape((-1,2))
            else:
                x_outs_true_trim = x_outs_true[:,burn_in_:,:x_dim_permuted_].reshape((-1,n_agents,n_dim_each_permuted)).reshape((-1,n_dim_each_permuted))
                x_outs_trim = x_outs[:,burn_in_:,:x_dim_permuted_].reshape((-1,n_agents,n_dim_each_permuted)).reshape((-1,n_dim_each_permuted))
            rmse_x = mean_squared_error(x_outs_true_trim.T,x_outs_trim.T, multioutput='raw_values',squared=False)
            
            
            print('RMSE_y: {:.4f} (+/-) {:.4f}\tRMSE_x: {:.4f} (+/-) {:.4f}\n'.format(
                np.mean(rmse),np.std(rmse),np.mean(rmse_x),np.std(rmse_x)))
            result = [rmse,rmse_x]
            if 'nba' in args.data:
                loss_all = np.mean(rmse) + 10*np.mean(rmse_x)
        else:
            print('RMSE_y: {:.4f} (+/-) {:.4f}\n'.format(np.mean(rmse),np.std(rmse)))
            result = [rmse]
        rmse_max,rmse_best_timing = None,None
        
    if save:
        res = [result, outcomes, covariates, others]
        with open(args.save_results+'.pkl', 'wb') as f:
            pickle.dump(res, f, protocol=4) 
        print(args.save_results+'.pkl is saved')
    return rmse_max,rmse_best_timing,rmse,loss_all


# MAIN
if __name__ == '__main__':
    
    # ---------------------------------------- # 
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30, required=True,
                        metavar='EPOC', help='train epochs')
    parser.add_argument('--batch-size', type=int, default=128, 
                        metavar='BS',help='batch size')
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--cuda-device', default=0, type=int, metavar='N',
                        help='which GPU to use')

    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--model', type=str, default='GDSW')
    parser.add_argument('--n_games', type=int, default=5)
    parser.add_argument('--val_devide', type=int, default=10)
    parser.add_argument('--numProcess', type=int, default=16)
    parser.add_argument('--syn_opt', type=str, default='')
    parser.add_argument('--gamma_p', type=float, default=0.3)
    parser.add_argument('--lambda_KL','--l_KL', type=float, default=0.1)
    parser.add_argument('--lambda_X','--l_X', type=float, default=0.1)
    parser.add_argument('--lambda_weight','--l_wt', type=float, default=0.1)
    parser.add_argument('--lambda_event','--l_ev', type=float, default=0.1)
    parser.add_argument('--small', action='store_true')
    parser.add_argument('--vel', action='store_true')
    parser.add_argument('--TEST', action='store_true')
    parser.add_argument('--cont', action='store_true')
    # parser.add_argument('--small', action='store_true')

    args = parser.parse_args()

    numProcess = args.numProcess  
    os.environ["OMP_NUM_THREADS"]=str(numProcess) 
    
    # Settings
    seed = 666
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if CUDA:
        torch.cuda.set_device(args.cuda_device)
        torch.cuda.manual_seed(seed)

    if args.data == 'nba':
        args.has_GT = False
        args.variable_length = True
    elif 'boid' in args.data :
        args.has_GT = True
        args.variable_length = False
    elif 'carla' in args.data:
        args.has_GT = True
        args.variable_length = True

    
    str_small = '_small' if args.small else ''
    str_vel = '_vel' if args.vel else ''
    str_param = '_w_'+str(args.lambda_weight)+'_X_'+str(args.lambda_X)+'_KL_'+str(args.lambda_KL)+'_event_'+str(args.lambda_event)

    args.save_model = './checkpoint/'+args.model+str_param+str_small+str_vel+'-'+args.data+'-'+str(args.n_games)+'.pt'
    args.save_results = './results/'+args.model+str_param+str_small+str_vel+'-'+args.data+'-'+str(args.n_games)
    print("save_model ==> ", args.save_model)

    shuffle=False
    args.lr = 1e-3

    # data loading
    if args.data == 'nba':
        data_dir = './datasets/TGV-CRN-nba'
        n_games = args.n_games
        val_devide = args.val_devide
        gamedata_dir = data_dir+'data/all_attacks2_nba_games_'+str(n_games)+'_VTEP_val5Fs10'
        gamedata_dir += '_vel/' # if args.vel else '/'
        args.n_agents = 11
        args.x_dim_permuted = args.n_agents*4 if not args.vel else args.n_agents*4 # 5 # 44
        args.x_dim_predicted = args.n_agents*2+1 if not args.vel else args.n_agents*2+1 #3 22
        
        args.burn_in = 95 # 95
        args.burn_in_test = 85#80#70
        args.burn_in0 = 10
        args.observation_window = 105#110#120
        args.max_v = 7

        if n_games == 5:
            args.batchsize_data = 64
        else:
            args.batchsize_data = 512
        gamedata_dir = gamedata_dir + '_batch' + str(args.batchsize_data) 
        args.gamedata_dir = gamedata_dir
        with open(gamedata_dir+'_tr_'+str(0)+'.pkl', 'rb') as f:
            data_batch,ind_tr,ind_te,_,_,_,_ = np.load(f,allow_pickle=True) 

        ind_tr_ = np.arange(ind_tr.shape[0])
        test_iids = np.arange(ind_te.shape[0])

        train_iids, val_iids,_,_ = train_test_split(ind_tr_, ind_tr_, test_size=1/val_devide, random_state=42)

        n_X_features = data_batch[0].shape[1]-51+1 if not args.vel else data_batch[0].shape[1]-51+1 # +args.n_agents
        n_X_static_features = 1
        n_X_fr_types = 1 # treatment
        n_classes = 1 # outcome

        # Datasets
        train_dataset = data_loader_nba.nbaDataset(train_iids, args.observation_window, args, TEST = False)
        val_dataset = data_loader_nba.nbaDataset(val_iids, args.observation_window, args, TEST = False)
        test_dataset = data_loader_nba.nbaDataset(test_iids, args.observation_window, args, TEST = True)
    
    elif args.data == 'carla':
        args.data_dir = './datasets/TGV-CRN-carla'
        args.burn_in0 = 10
        args.burn_in = 39 # 
        args.burn_in_test = 39
        args.t_eliminated = 5#
        args.observation_window = 60 #  
        args.n_agents = 10 #
        args.x_dim_permuted = args.n_agents*7 # 
        args.x_dim_predicted = args.n_agents*3 # 
        args.n_agents_all = 41 # 
        obs_w = args.observation_window
        args.t_future_waypoint = 10
        args.t_future = 1
        args.max_p = 250
        args.max_v = 15
        args.max_y = 1000
        n_X_features = args.x_dim_permuted+4 #
        n_X_static_features = 1
        n_X_fr_types = 1 # treatment
        n_classes = 1 # outcome
        args.vel = True
        # Datasets
        vacant_crowd = ['crowd','vacant']
        intervention = ['int','noint']

        metadata_path = args.data_dir+'metadata2'
        filename_all = glob.glob(args.data_dir+'*.pickle')
        filename_all2 = [filename_all[f].replace(args.data_dir,'') for f in range(len(filename_all))]

        i_file = 0
        if not os.path.isfile(metadata_path+'.pickle'):  
            IDs, types, mileage_progresses, mileages, collisions,filenames,filenames_shorter,filenames_no,filelen,interventions = [],[],[],[],[],[],[],[],[],[]
            for town in range(1,8):
                for vc in range(2):
                    for ID in range(100): #40 20
                        for ID2 in range(15):
                            for i in range(2):
                                filename = 'town0{}_{}_{}_{}_{}.pickle'.format(town,vacant_crowd[vc],ID,intervention[i],ID2)
                                flag_data = False
                                try: 
                                    data = np.load(args.data_dir + filename)
                                    flag_data = True
                                except: 
                                    if filename in filename_all2:
                                        filenames_no.append(filename)
                                        import pdb; pdb.set_trace()
                                    else:
                                        print(filename+' does not exist ')
                                if flag_data:
                                    i_file += 1 
                                    mileage_progress,mileage,collision,interv = [],[],[],[]
                                    # for dat in data['waypoint']:
                                    t_int = 0

                                    for dat in data['drive_data']:
                                        try: IDs.append([*dat['actors'].keys()])
                                        except: import pdb; pdb.set_trace()
                                        mileage_progress.append(dat['mileage_progress'])
                                        mileage.append(dat['mileage'])
                                        collision.append(dat['collision'])
                                        
                                        try: 
                                            IDs[-1].remove('ego_vehicle')
                                        except:
                                            print('ego_vehicle is not found')
                                        IDs[-1] = np.array(IDs[-1])
                                        type_ = []
                                        for id in IDs[-1]:
                                            types.append(dat['actors'][id]['type'])
                                            type_.append(dat['actors'][id]['type'])
                                        if len(type_) < len(np.unique(type_)):
                                            import pdb; pdb.set_trace()
                                        interv.append(dat['intervention'])
                                    tmp_interv = np.array(interv).astype(np.int)
                                    if i == 0:
                                        tmp_interv0 = tmp_interv
                                        

                                    mileage_progresses.append(np.array(mileage_progress))
                                    mileages.append(np.array(mileage))
                                    continous = np.sum(np.array(mileage_progress)==mileage_progress[-1])
                                    collision1st = np.where(np.array(collision)==1)[0]
                                    collision1st = collision1st[0] if len(collision1st)>0 else 999
                                    collisions.append(collision1st)
                                    collision1st = str(collision1st) if collision1st<999 else 'no'
                                    filelen.append(len(mileage_progress))

                                    # 'town '+str(town)+' ID '+str(ID)+' vacant_crowd '+str(vc)+' interv '+str(i)
                                    
                                    print(filename+' (total'+str(i_file)+') 1st_collision '+ str(collision1st)
                                        +' len '+str(len(mileage_progress))+' last '+str(mileage_progress[-1])+' continous '+str(continous))
                                    
                                    if i == 1:
                                        eliminated = args.observation_window + args.t_eliminated + args.t_future_waypoint
                                        minlength = np.min([len(mileage_progresses[-1]),len(mileage_progresses[-2])])
                                        try:
                                            if minlength >= eliminated and \
                                                np.mean(np.abs(mileages[-1][:65]-mileages[-2][:65])) < 10 and \
                                                (mileages[-1][10]-mileages[-1][0]>0) and (mileages[-2][10]-mileages[-2][0]>0) and \
                                                (np.min(mileages[-1][6:]-mileages[-1][5])>0) and (np.min(mileages[-2][6:]-mileages[-2][5])>0):
                                                # collisions[-1] >= args.burn_in+args.t_eliminated and collisions[-2] >= args.burn_in+args.t_eliminated:
                                                filenames.append(filename_prev)
                                                filenames.append(filename)
                                                try: interventions.append(np.where(tmp_interv0==1)[0][0])
                                                except: import pdb; pdb.set_trace()
                                            else:
                                                print('shorter or inappropreate data')
                                                filenames_shorter.append(filename_prev)
                                                filenames_shorter.append(filename)

                                        except: import pdb; pdb.set_trace()
                                    else:
                                        filename_prev = filename
                                    
                                

            ID_all = np.unique(np.concatenate(IDs))
            types_all = np.unique(types)
            
            
            metadata = [ID_all, types_all, filenames, filelen, interventions]
            with open(metadata_path+'.pickle', 'wb') as f:
                pickle.dump(metadata, f, protocol=4) 
        else:
            with open(metadata_path+'.pickle', 'rb') as f:
                ID_all, types_all, filenames, filelen, interventions = np.load(f,allow_pickle=True) 

        args.ID_all = ID_all
        args.types_all = types_all
        args.filenames = filenames

        args.n_samples = 2 
        factual = np.array([random.randrange(args.n_samples) for _ in range(int(len(args.filenames)/args.n_samples))])
        
        # for check
        #result = [i for i in filename_all2 if i not in filenames]
        #result = sorted(result)

        # intervention_shift
        # np.min(np.array(filelen) # min=86
        # print(interventions)
        time_shift = np.array([random.randrange(10,20) for _ in range(int(len(args.filenames)/args.n_samples))])
        time_shift += np.array(interventions)-65       

        train_ratio = 0.7
        val_ratio = 0.1
        len_file = len(filenames)/2/10 if args.small else len(filenames)/2

        all_idx = np.arange(int(len_file))
        np.random.shuffle(all_idx)
        train_iids = all_idx[:int(len(all_idx)*train_ratio)]
        val_iids = all_idx[int(len(all_idx) * train_ratio):int(len(all_idx) * train_ratio)+int(len(all_idx) * val_ratio)]
        test_iids = all_idx[int(len(all_idx) * train_ratio)+int(len(all_idx) * val_ratio):]
        factual_train = factual[train_iids] ; factual_val = factual[val_iids] ; factual_test = factual[test_iids] 
        
        print('All/train/valid/test files: '+str(int(len(filenames)/2))+'/'+str(len(train_iids))+'/'+str(len(val_iids))+'/'+str(len(test_iids)))
        train_dataset = data_loader_carla.carlaDataset(train_iids, args.observation_window, factual_train, time_shift, args)
        val_dataset = data_loader_carla.carlaDataset(val_iids, args.observation_window, factual_val, time_shift, args, TEST = True)
        test_dataset = data_loader_carla.carlaDataset(test_iids, args.observation_window, factual_test, time_shift, args, TEST = True)

    elif 'boid' in args.data:
        args.n_agents = 20
        args.x_dim_permuted = args.n_agents*7
        args.x_dim_predicted = args.n_agents*2
        args.burn_in0 = 3#5
        args.burn_in = 9#19
        args.burn_in_test = 9#19
        args.observation_window = 16# 29/26
        args.intervention_window = 5
        args.N_data = 20000 if args.data == 'boid' else 20000
        args.N_data_test = 400
        args.max_p = 15
        args.vel = True

        train_iids = np.arange(args.N_data) 
        val_iids = np.arange(args.N_data_test) 
        test_iids = np.arange(args.N_data_test)
        factual_val = np.random.choice(range(args.intervention_window+1),args.N_data_test) # -args.burn_in
        factual_test = np.random.choice(range(args.intervention_window+1),args.N_data_test)

        train_dataset = data_loader_boid.boidDataset(train_iids, args.observation_window, args, train = 1)
        val_dataset = data_loader_boid.boidDataset(val_iids, args.observation_window, args, train = 0, factual_test=factual_val)
        test_dataset = data_loader_boid.boidDataset(test_iids, args.observation_window, args, train = -1, factual_test=factual_test)

        n_X_features = args.x_dim_permuted +1 
        n_X_static_features = 1
        n_X_fr_types = 1 # treatment
        n_classes = 1 # outcome
    
    if 'boid' in args.data or args.data == 'carla':
        import scipy.io as sio
        sio.savemat('mat/'+args.data+'_factual_test.mat',{'factual_test':factual_test}) #

    # DataLoaders
    kwargs = {'num_workers': 8, 'pin_memory': True} if CUDA else {} #  {} # 
    mp.set_start_method('spawn')
    train_loader = torch.utils.data.DataLoader(train_dataset, args.batch_size, shuffle=shuffle, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, args.batch_size, shuffle=shuffle, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, args.batch_size, shuffle=shuffle, **kwargs)

    args.y_pred = False if "NP" in args.model else True # 

    print("batch size ==> ", args.batch_size )
    print("lr ==> ", args.lr)
    print("data ==> ", args.data)
    print("model ==> ", args.model)
    print("y_pred ==> ", args.y_pred)
    # print("observation window == >", args.observation_window)
    if args.data == 'carla':
        args.y_positive = True
        args.x_residual = False if "T" in args.model else True # 
        args.dim_rec_global = 1
        args.y_residual = True
        args.rollout_y_train = False
    elif 'boid' in args.data:
        args.y_positive = True # False
        args.x_residual = False
        args.y_residual = False # True
        args.dim_rec_global = 1
        args.rollout_y_train = False
    elif args.data == 'nba':
        args.y_positive = True
        args.x_residual = False if args.vel and "T" in args.model else True
        args.y_residual = False
        args.dim_rec_global = 6 # 4 # 11
        args.rollout_y_train = False
    # ---------------------------------------- # 
    # Model 
    
    attn_model = 'concat2'
    n_Z_confounders = HIDDEN_SIZE
    

    if 'V' in args.model:
        model = GVCRN(n_X_features, n_X_static_features,
                n_classes, args, hidden_size = HIDDEN_SIZE)    
    elif 'RNN' in args.model:
        model = RNN(n_X_features, n_X_static_features, 
                n_classes, args, hidden_size = HIDDEN_SIZE)   
    else:
        model = GDSW(n_X_features, n_X_static_features, n_X_fr_types, n_Z_confounders,
                attn_model, n_classes, args, hidden_size = HIDDEN_SIZE)

    adam_optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print('lambda weights:'+str(args.lambda_weight)+' X:'+str(args.lambda_X)+' KL:'+str(args.lambda_KL)+' event:'+str(args.lambda_event))
    trainInitIPTW(train_loader, val_loader,test_loader,
                  model, args, epochs= args.epochs,
                  criterion=F.binary_cross_entropy_with_logits, optimizer=adam_optimizer,
                  use_cuda=CUDA,
                  save_model=args.save_model, TEST=args.TEST)
