import numpy as np
import random
import matplotlib.pyplot as plt
import time
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
# from p5 import Vector, stroke, circle
import warnings
warnings.simplefilter('error')

class Boid(object):
    def __init__(self, n_boids, width, delta_T, r_orientation= 10, noise_var=0.0):
        # self.max_force = 0.3
        self.speed = 1
        self.r_repulsion = 0.5 # 2 
        # self.r_repulsion2 = 5
        self.r_orientation = r_orientation # 5/10
        self.r_attraction = 7.5 # 10 (all) default:15 
        self.n_boids = n_boids
        self._delta_T = delta_T
        self.beta = 0.8727*0.6*self._delta_T # Maximum turning angle # 3*np.pi#

        self.width = width
        self.noise_var = noise_var
        # self.length = 1 # length of boids 0.5
        
    '''def __init__(
        self,
        
        box_size=5.0,
        loc_std=0.5,
        vel_norm=0.5,
        noise_var=0.0,
    ):
        self.n_boids = n_boids
        self.box_size = box_size
        self.loc_std = loc_std
        self.vel_norm = vel_norm
        self.noise_var = noise_var

        self._boid_types = np.array([0.0, 0.5, 1.0])
        
        # self._max_F = 0.1 / self._delta_T'''


    def _clamp(self, loc, vel):
        """
        :param loc: 2xN location at one time stamp
        :param vel: 2xN velocity at one time stamp
        :return: location and velocity after hitting walls and returning after
            elastically colliding with walls
        """
        assert np.all(loc < self.width * 3)
        assert np.all(loc > -self.width * 3)

        over = loc > self.width
        loc[over] = 2 * self.width - loc[over]
        assert np.all(loc <= self.width)

        # assert(np.all(vel[over]>0))
        vel[over] = -np.abs(vel[over])

        under = loc < -self.width
        loc[under] = -2 * self.width - loc[under]
        # assert (np.all(vel[under] < 0))
        assert np.all(loc >= -self.width)
        vel[under] = np.abs(vel[under])

        return loc, vel
 
    def get_edges(self,args):
        if args.partial and args.avoid:
            edges = np.random.choice(
                np.array([-1., 0., 1.]), size=(self.n_boids, self.n_boids), p=np.array([1/5, 2/5, 2/5])
            )
        elif args.partial and not args.avoid:
            ratio = 3/4 ; value_neg = 0 
            edges = np.random.choice(
                np.array([value_neg, 1.0]), size=(self.n_boids, self.n_boids), p=np.array([1-ratio, ratio])
            )
        else:
            edges = np.ones((self.n_boids, self.n_boids))

        np.fill_diagonal(edges, 0)
        return edges

    def sample_trajectory(
        self,
        args, i_sim,
        T=10000,
        sample_freq=1,
        edges=None,
    ):
        n = self.n_boids
        assert T % sample_freq == 0
        # T_save = int(T / sample_freq )
        T_save = int(T / sample_freq - 1)

        # change r_o
        self.r_o2 = 4
        burn_in = 20 if args.ver == 1 else 10
        burn_in_ = burn_in
        Tc_range = 5
        Tc_range_ = Tc_range
        T_zero = 1 
        if args.train==1: # 
            burn_in = np.random.choice(range(5,burn_in_+Tc_range+1))
            #burn_in = np.random.choice(range(burn_in,burn_in+Tc_range+2))
            if burn_in >= burn_in_+Tc_range:
                burn_in = 999
            Tc_range = 1; T_zero = 0

        if edges is None or args.bat:
            edges = self.get_edges(args)
        sd = 0.2
        speed = self.speed * sd*np.random.rand(n) 
        r_r = self.r_repulsion + sd*np.random.rand(n) 
        # r_r2 = self.r_repulsion2 + sd*np.random.rand(n) 
        r_r2 = None
        if self.r_repulsion == self.r_orientation:
            r_o1 = r_r
        else:
            r_o1 = self.r_orientation + sd*np.random.rand(n)/10
        r_o2 = r_o1 + (self.r_o2 - self.r_orientation) # + sd*np.random.rand(n)/10 
        r_a = self.r_attraction + sd*np.random.rand(n) 
        beta = self.beta

        # Initialize location and velocity
        loc = np.zeros((Tc_range+T_zero,T_save, 2, n))
        vel = np.zeros((Tc_range+T_zero,T_save, 2, n))
        phis = np.zeros((Tc_range+T_zero,T_save, 1, n))
        edges_res = np.zeros((Tc_range+T_zero,T_save, n, n))
        edges_all = np.zeros((Tc_range+T_zero,T, n, n))
        treatment = np.zeros((Tc_range+T_zero,T_save, 1))
        if args.train==1:
            if burn_in < 999:
                treatment[0,burn_in:] = 1
        else:
            for t in range(Tc_range):
                if t <= Tc_range-1:
                    treatment[t,burn_in+t:] = 1
            # treatment[:,:,0].T

        rand_p = np.random.randn(1, n) * 2* np.pi
        loc_next_ = (3 + 3 * np.random.rand(2, n)) * np.array([np.cos(rand_p),np.sin(rand_p)]).squeeze() # 6
        try: vel_next_ = np.array([-np.sin(rand_p),np.cos(rand_p)]).squeeze() * self.speed  # speed
        except: import pdb; pdb.set_trace()
        loc_next_ += vel_next_
        phi_next = rand_p
        # loc[0, :, :], vel[0, :, :] = self._clamp(loc_next, vel_next)
        res_prev = np.zeros((n,n))

        # run 
        ttt = 0; angvel_prev = 0
        #if args.train==1:
        #    import pdb; pdb.set_trace()
        #else:
        start_tt = burn_in
        end_tt = burn_in+Tc_range+T_zero

        for tt in range(start_tt,end_tt):#+1
            loc_next = loc_next_.copy()
            vel_next = vel_next_.copy()

            counter = 0
            counter2 = 0
            r_o = r_o1

            for t in range(1, T):
                if t >= tt and tt < burn_in_+Tc_range_:
                    r_o = r_o2

                res = np.zeros((n,n))

                loc_next, vel_next = self._clamp(loc_next, vel_next)
                
                if t % sample_freq == 0:
                    loc[ttt,counter, :, :], vel[ttt,counter, :, :] = loc_next, vel_next
                    phis[ttt,counter, :, :] = phi_next
                    counter += 1
                    

                # apply_behavior 
                di = np.zeros((2,n))
                vel_ = vel_next.copy()
                for i in range(n):
                    di[:,i], repul_flag, res = self.repulsion(loc_next, vel_next, i, edges, r_r, r_r2, res, args)

                    if not repul_flag:
                        di[:,i], oa_flag, res = self.orient_attract(loc_next, vel_next, i, edges, r_r, r_o, r_a, res, res_prev, args)
                    
                    # turn angle limitation
                    signum = np.sign(vel_next[0,i]*di[1,i]-di[0,i]*vel_next[1,i]) # Compute direction of the angle
                    dotprod = np.dot(di[:,i],vel_next[:,i])  # Dotproduct (needed for angle)
                    if np.linalg.norm(di[:,i]) > 1e-10:
                        cos_theta = dotprod/np.linalg.norm(di[:,i])/np.linalg.norm(vel_next[:,i])
                        if np.abs(cos_theta) <= 1: 
                            try: phi = np.real(signum*np.arccos(cos_theta)) # Compute angle
                            except: import pdb; pdb.set_trace()
                        else: 
                            phi = 0.0
                    else: 
                        phi = 0.0
                    
                    # phi += 0.001*np.random.rand() 
                    

                    try: 
                        if abs(phi) <= beta:                                                    
                            vel_[:,i] = np.matmul(np.array([[np.cos(phi),-np.sin(phi)],[np.sin(phi),np.cos(phi)]]),vel_next[:,i])           
                            phi_next = phi
                        elif phi < beta:                                                                  
                            vel_[:,i] = np.matmul(np.array([[np.cos(-beta),-np.sin(-beta)],[np.sin(-beta),np.cos(-beta)]]),vel_next[:,i])
                            phi_next = -beta
                        else:                                                                                                
                            vel_[:,i] = np.matmul(np.array([[np.cos(beta),-np.sin(beta)],[np.sin(beta),np.cos(beta)]]),vel_next[:,i])
                            phi_next = beta
                    except: import pdb; pdb.set_trace()

                # update
                loc_next += vel_next*self._delta_T 
                vel_next = vel_ 
                

                try: edges_res[ttt,counter2] += res/sample_freq
                except: import pdb; pdb.set_trace()
                edges_all[ttt,t] = res

                res_prev = res.copy()
                if t % sample_freq == sample_freq-1 and t > sample_freq:
                    counter2 += 1


            center = np.mean(loc[ttt],2)
            vec_ic = loc[ttt] - center[:,:,np.newaxis].repeat(args.n_boids,2)
            angvel = np.cross(vec_ic,vel[ttt],axis=1)
            try: angvel = np.mean(angvel[burn_in_+Tc_range_:])
            except: import pdb; pdb.set_trace()
            if tt < burn_in_+Tc_range_ and burn_in < 999:
                print("intervention: {}, Ang vel: {:.3f}".format(tt, angvel))
                # print('intervention: '+str(tt))
            else:
                print("no intervention: {}, Ang vel: {:.3f}".format(tt, angvel))
            
            
            #if tt >=21: #and args.train <= 0: 
            #    import pdb; pdb.set_trace()


            ttt += 1 
            angvel_prev = angvel
        edges_result = edges
        #if args.train == 0:
        #    import pdb; pdb.set_trace()
        return loc, vel, phis, edges, treatment

    def repulsion(self, loc, vel, i, edges, r, r2, res, args):
        total = 0
        avg_vector = np.zeros(2)
        for j in range(self.n_boids):
            if args.avoid:
                flag = (edges[i,j] != 0)
                r_ = r if edges[i,j] == 1 else r2
            else:
                flag = (edges[i,j] == 1) # (i != j) if args.avoid_all else
                r_ = r
            if flag: 
                distance = np.linalg.norm(loc[:,j] - loc[:,i])
                if distance < r_[i]:
                    diff = loc[:,j] - loc[:,i]
                    diff /= distance
                    avg_vector += diff
                    total += 1
                    res[i,j] += -1

        if total > 0:
            steering = -avg_vector/total
            repul_flag = True
        else: 
            steering = avg_vector
            repul_flag = False

        return steering, repul_flag, res


    def orient_attract(self, loc, vel, i, edges, r_r, r_o, r_a, res, res_prev, args):
        
        total_o = 0
        avg_vector = np.zeros(2)
        total_a = 0
        center_of_mass = np.zeros(2)

        for j in range(self.n_boids):
            if edges[i,j] == 1: 
                try: dist = np.linalg.norm(loc[:,j] - loc[:,i])
                except: import pdb; pdb.set_trace()
                if dist >= r_r[i] and dist < r_o[i]: # orientation
                    avg_vector += vel[:,j]
                    total_o += 1
                    if res_prev[i,j]==1: # attraction -> orientation
                       res[i,j] += -0.5
                    elif res_prev[i,j]==-1: # repulsion -> orientation
                        res[i,j] += 0.5
                    else: # initial or continuing or unknown
                        res[i,j] += 0

                elif dist >= r_o[i] and dist < r_a[i]: # attraction
                    center_of_mass += loc[:,j]
                    total_a += 1
                    res[i,j] += 1


        if total_o > 0:
            steering_o = avg_vector/total_o
        else:
            steering_o = avg_vector

        if total_a > 0:
            center_of_mass /= total_a
            vec_to_com = center_of_mass - loc[:,i]
            if np.linalg.norm(vec_to_com) > 0:
                vec_to_com = (vec_to_com / np.linalg.norm(vec_to_com)) 
            steering_a = vec_to_com
        else:
            steering_a = np.zeros(2)

        if total_o > 0 and total_a > 0:
            steering = (steering_o + steering_a)/2
        else:
            steering = steering_o + steering_a

        return steering, (total_o + total_a), res

