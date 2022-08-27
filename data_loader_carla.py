import numpy as np
import torch
from torch.utils import data

class carlaDataset(data.Dataset):
    def __init__(self, list_IDs, obs_w, factual, time_shift, args, TEST=False):
        '''Initialization'''
        self.list_IDs = list_IDs
        self.obs_w = obs_w
        self.data_dir = args.data_dir
        self.ID_all = args.ID_all
        self.types_all = args.types_all
        self.filenames = args.filenames

        self.burn_in = args.burn_in
        self.t_eliminated = args.t_eliminated
        self.time_shift = time_shift
        self.x_dim_permuted = args.x_dim_permuted
        self.n_agents = args.n_agents
        self.n_agents_all = args.n_agents_all
        self.t_future = args.t_future
        self.t_future_waypoint = args.t_future_waypoint
        self.n_samples = args.n_samples 
        self.factual = factual
        self.TEST = args.TEST
        self.max_p = args.max_p
        self.max_v = args.max_v
        self.max_y = args.max_y        
    def __len__(self):
        '''Denotes the total number of samples'''
        return len(self.list_IDs)

    def __getitem__(self, index):
        '''Generates one sample of data'''
        data_dir = self.data_dir
        # Select sample
        id = self.list_IDs[index]
        n_samples = self.n_samples 
        factual = self.factual[index]
        t_eliminated = self.t_eliminated + self.time_shift[index]

        speeds_, sizes_, types_, poses_,lengths = [],[],[],[],[]
        lengths = np.zeros((n_samples,))
        results = np.zeros((self.obs_w,n_samples))
        interventions = np.zeros((self.obs_w+1,n_samples))
        intervention_targets = np.zeros((self.obs_w,3,n_samples))
        mileages = np.zeros((self.obs_w,n_samples))
        collisions = np.zeros((self.obs_w,n_samples))
        waypoints = np.zeros((self.obs_w,2*self.t_future_waypoint,n_samples)) # 4

        for n in range(n_samples):
            # Load data
            data = np.load(data_dir + self.filenames[n_samples*id+n])
            intervention,intervention_target,collision,mileage,mileage_progress,simulate_progress,time,IDs= [],[],[],[],[],[],[],[]
            waypoint = []

            for dat in data['drive_data']:
                intervention.append(dat['intervention'])
                if len(dat['intervention_target']) == 0:
                    intervention_target.append(np.zeros(3))
                else:
                    intervention_target.append(dat['intervention_target'])
                collision.append(dat['collision'])
                mileage.append(dat['mileage'])
                time.append(dat['time'])
                IDs = [*dat['actors'].keys()]

            for dat in data['waypoint']:
                waypoint.append(np.array([dat['x'],dat['y'],dat['z'],dat['yaw']]))
            
            IDs = np.unique(np.array(IDs))
            # IDs = np.concatenate([IDs[-1,np.newaxis,np.newaxis],IDs[:-1,np.newaxis]],0).squeeze(1)
            intervention = np.array(intervention).astype(np.int)
            intervention_target = np.array(intervention_target)
            collision = np.array(collision).astype(np.int)
            collision[:t_eliminated+self.burn_in] = 0 # ignore collision
            mileage = np.array(mileage)
            waypoint = np.array(waypoint)
            time = np.array(time)-time[0]


            # for each object
            n_all = len(self.types_all)
            speeds = np.zeros((self.obs_w,n_all,1))
            poses = np.zeros((self.obs_w,n_all,4))
            sizes = np.zeros((self.obs_w,n_all,3))
            waypoint_ = np.zeros((self.obs_w,2*self.t_future_waypoint)) # 4
            
            try: types = [['' for _ in range(n_all)] for _ in range(self.obs_w)]
            except: import pdb; pdb.set_trace()
            
            i = 0; t = 0 
            mileage_decrease = 0
            strictly_safe = False
            for dat in data['drive_data']:
                if collision[t] == 1:                
                    if strictly_safe:
                        # collision[t:] = 1 
                        continue
                    else:
                        mileage_decrease += mileage[t]-mileage[t-1]
                if t >= t_eliminated and i < self.obs_w : 
                    if strictly_safe: 
                        if np.sum(collision[t+1:t+self.t_future+1])==0:
                            results[i,n] = mileage[t+self.t_future]
                    else:
                        results[i,n] = mileage[t+self.t_future] - mileage_decrease

                    for ID in IDs:
                        if ID != 'ego_vehicle':
                            ID = int(ID)
                        else:
                            ID_ego = ID
                            
                        if ID in [*dat['actors'].keys()]:
                            ind = list(self.types_all).index(dat['actors'][ID]['type'])
                            speeds[i,ind] = dat['actors'][ID]['speed']
                            poses[i,ind] = np.array(dat['actors'][ID]['pose'])
                            sizes[i,ind] = np.array(dat['actors'][ID]['size'])
                            types[i][ind] = dat['actors'][ID]['type']
                            
                        if ID == 'ego_vehicle':
                            nearest_waypoints = np.argmin(np.sum((waypoint[:,:2]-poses[np.newaxis,i,ind,:2].repeat(waypoint.shape[0],0))**2,1))
                            T1 = nearest_waypoints+self.t_future_waypoint
                            T0 = waypoint.shape[0]
                            if waypoint.shape[0]<nearest_waypoints+1:
                                import pdb; pdb.set_trace()
                            elif waypoint.shape[0]<T1:
                                T2 = self.t_future_waypoint-T1+T0# T1-T0
                                try: waypoint_[i,:T2*2] = waypoint[nearest_waypoints:T1,:2].reshape((-1,))
                                except: import pdb; pdb.set_trace()
                                waypoint_[i,T2*2:] = np.repeat(waypoint[T0-1:T0,:2],self.t_future_waypoint-T2,axis=0).reshape((-1,))
                            else:
                                waypoint_[i] = waypoint[nearest_waypoints:T1,:2].reshape((-1,))

                    i += 1
                t += 1

            ind_ego = list(self.types_all).index(dat['actors'][ID_ego]['type'])          
            speeds = np.concatenate([speeds[:,ind_ego:ind_ego+1],speeds[:,:ind_ego],speeds[:,ind_ego+1:]],1)
            poses = np.concatenate([poses[:,ind_ego:ind_ego+1],poses[:,:ind_ego],poses[:,ind_ego+1:]],1)
            sizes = np.concatenate([sizes[:,ind_ego:ind_ego+1],sizes[:,:ind_ego],sizes[:,ind_ego+1:]],1)

            dist = np.sqrt(np.sum((poses[:,0:1,:2].repeat(self.n_agents_all-1,1)-poses[:,1:,:2])**2,2))
            ind_dist = np.argsort(np.min(dist,axis=0),axis=0)
            ind_dist = np.concatenate([np.zeros((1,)),ind_dist+1],0).astype(np.int32)

            speeds__ = speeds[:,:self.n_agents].copy()
            poses__ = poses[:,:self.n_agents].copy()
            sizes__ = sizes[:,:self.n_agents].copy()
            for t in range(self.obs_w):
                try: speeds__[t] = speeds[t,ind_dist[:self.n_agents]] # t,
                except: import pdb; pdb.set_trace()
                poses__[t] = poses[t,ind_dist[:self.n_agents]] # t,
                sizes__[t] = sizes[t,ind_dist[:self.n_agents]] # t,
            speeds = speeds__
            poses = poses__
            sizes = sizes__
                
            lengths[n] = i 
            interventions[:,n] = intervention[t_eliminated:t_eliminated+self.obs_w+1]
            intervention_targets[:,:,n] = intervention_target[t_eliminated:t_eliminated+self.obs_w]
            collisions[:,n] = collision[t_eliminated:t_eliminated+self.obs_w]
            if n == 0:
                interventions[np.where(interventions[:,0]==1)[0][0]-1:,0] = 1
            waypoints[:,:,n] = waypoint_
            
            
            mileages[:,n] = mileage[t_eliminated:t_eliminated+self.obs_w]
            results[:,n] -= np.repeat(mileages[0,np.newaxis,n],self.obs_w)
            mileages[:,n] -= np.repeat(mileages[0,np.newaxis,n],self.obs_w) 
            if (results[10,n]-results[0,n]>0) and np.min(results[10:,n])<0.01:
                import pdb; pdb.set_trace() 
            if i < self.obs_w:
                speeds[i:] = speeds[i-1:i].repeat(self.obs_w-i,0)
                poses[i:] = poses[i-1:i].repeat(self.obs_w-i,0)
                waypoint_[i:] = waypoint_[i-1:i].repeat(self.obs_w-i,0)
                mileages[i:] = mileages[i-1:i].repeat(self.obs_w-i,0)

            speeds_.append(np.array(speeds))
            poses_.append(np.array(poses))      
            types_.append(np.array(types))
            sizes_.append(np.array(sizes))
        speeds_ = np.array(speeds_)/self.max_v
        poses_ = np.array(poses_)
        types_ = np.array(types_)
        sizes_ = np.array(sizes_)
        lengths = np.array(lengths)

        poses_ = np.concatenate([poses_[:,:,:,:2]/self.max_p,np.cos(poses_[:,:,:,3:4]),np.sin(poses_[:,:,:,3:4])],3) # x,y,cos,sin (eliminate z)
        sizes_ = sizes_[:,:,:,:2] # x,y  

        #if not self.TEST:
        lengths = lengths[factual]
        
        # output
        X_demographic = np.concatenate([np.zeros((1)),np.max(np.max(sizes_,1),0).reshape((-1,))],0) # 1(dummy)+agent*3
        X_demo = torch.from_numpy(X_demographic.astype(np.float32)) 
        y_all = torch.from_numpy(results.astype(np.float32))/self.max_y
        y = y_all[:,factual]
        
        X_treatment_opt = np.argmax(y_all[-1,:])
        X_treatment_all = torch.from_numpy(interventions.astype(np.float32)) 
        X_treatment = X_treatment_all[:,factual]
        horizon = self.obs_w-self.burn_in

        X_ind = np.concatenate([speeds_,poses_,sizes_],3).transpose((1,2,3,0)).reshape((self.obs_w,7*self.n_agents,n_samples)) # n,t,agent,dim -> t,dim*agent,n # n_all
        X_all = np.concatenate([X_ind,mileages[:,np.newaxis]/self.max_y,X_ind[:,0:1,:],np.zeros((self.obs_w,1,2)),collisions[:,np.newaxis]],1) # 

        X_all = torch.from_numpy(X_all.astype(np.float32)) # t,dim_all,n
        X = X_all[:,:,factual] # 
        return X, X_demo, X_treatment, y, lengths, index, X_treatment_all, X_all, y_all, X_treatment_opt

