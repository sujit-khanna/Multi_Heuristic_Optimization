
'''
This file contains the implementation of Ensemble Optimizer
'''
import numpy as np
import math as m
import random
from PSO import *
from Differential_Evolution import *
from Cuckoo_Search import *
from Flower_Pollination import *
from objective_func import *
from PSOLevy import *
from Bat import *
from BatLevy import *
import statistics
import pandas as pd
import random
from matplotlib import pyplot as plt
import pandas as pd


class init_EO:

    def __init__(self, obj, obj_dim,pop_size, ceiling, floor, seed, **kwargs):
        self.obj_dim = obj_dim
        self.objective = obj
        self.num_fireflies = pop_size
        self.ceiling = ceiling
        self.floor = floor
        self.seed = seed
        self.init_pop = self.initialize_population()
        self.fitness = self.calc_fitness()

    def calc_fitness(self):
        tmp = self.init_pop
        tmp_1 = self.init_pop.shape
        pop_intensity = np.asarray(np.zeros(self.init_pop.shape[0])).astype(float)
        for i in range(len(pop_intensity)):
            pop_intensity[i] = self.objective(self.init_pop[i])
        return pop_intensity

    def initialize_population(self):
        random.seed(self.seed)
        pop_list = np.asarray(np.zeros((self.num_fireflies, self.obj_dim))).astype(float)
        for i in range(self.num_fireflies):
            pop_list[i] = np.array([random.uniform(self.floor, self.ceiling) for _ in range(self.obj_dim)])
        return pop_list



class EnsembleOptimizer:

    def __init__(self, obj, class_obj, num_agents, num_gen, seed, obj_dim, communicate,scheme, wts_decay, floor=-5, ceiling=5, **kwargs):
        self.obj=obj
        self.class_obj = class_obj
        self.num_agents=num_agents
        self.num_gen = num_gen
        self.seed = seed
        self.obj_dim = obj_dim
        self.scheme = scheme
        self.decay = wts_decay
        self.floor = floor
        self.ceiling = ceiling
        self.ind_best = None
        self.ensemble_best = None
        self.runner = None
        self.randomize = None
        self.fitness = None
        self.err_tracker = []
        self.DE=None
        self.PSO=None
        self.PSOL=None
        self.Cuckoo=None
        self.flower = None
        self.Bat=None
        self.BatL=None
        self.rank_weights = None
        self.final_best=None
        self.comms = communicate
        self.final_best_params = None
        self.loss_tracker = []

    def new_intensity(self, param):
        return self.obj(param)

    def init_optimization(self):
        self.runner = init_EO(self.obj, self.obj_dim, self.num_agents, self.ceiling, self.floor, self.seed)
        self.population=self.runner.init_pop
        self.fitness = self.runner.fitness

    def custom_sort(self, arr):
        l = list(arr)
        l.sort(key=lambda crunch: self.new_intensity(crunch))
        sort_ary = np.asarray(l)
        return sort_ary

    def meta_weighing_scheme(self, best_list):
        '''weighing schemes used 1.best, 2. average, 3. rank weighted, 4. exponential rank weiging'''
        weighted_best = self.custom_sort(best_list)
        best_param_1 = weighted_best[len(weighted_best)-1]

        weighted_best_2 = np.mean(best_list, axis=0)
        best_param_2 = weighted_best_2

        self.rank_weights = list(range(1,8,1))
        weighted_best_3 = self.custom_sort(best_list)
        best_param_3 = np.average(weighted_best_3, weights=self.rank_weights, axis=0)

        weighted_best_4 = self.custom_sort(best_list)
        self.rank_weights = list(range(7,0,-1))
        exp_wts = []
        for i in range(len(self.rank_weights)):
            tmp = self.rank_weights[i]*self.decay**abs((self.rank_weights[i] - self.rank_weights[0]))
            exp_wts.append(tmp)
        exp_np = np.array(exp_wts)
        exp_wts = exp_np/np.sum(exp_np)
        best_param_4 = np.average(weighted_best_4, weights=np.flip(exp_wts, axis=0), axis=0)
        a = best_param_1.shape[0]
        tmp_best = np.asarray(np.zeros((4, best_param_1.shape[0]))).astype(float)
        tmp_best[0,:], tmp_best[1,:], tmp_best[2,:], tmp_best[3,:]=best_param_1,best_param_2,best_param_3,best_param_4
        meta_param = np.mean(tmp_best, axis=0)

        return meta_param


    def weighing_scheme(self, best_list):
        '''weighing schemes used 1.best, 2. average, 3. rank weighted, 4. exponential rank weiging'''
        if self.scheme=='best':
            weighted_best = self.custom_sort(best_list)
            best_param = weighted_best[len(weighted_best)-1]
        elif self.scheme == 'average':
            weighted_best = np.mean(best_list, axis=0)
            best_param = weighted_best
        elif self.scheme == 'rank':
            self.rank_weights = list(range(1,8,1))
            weighted_best = self.custom_sort(best_list)
            best_param = np.average(weighted_best, weights=self.rank_weights, axis=0)

        elif self.scheme == 'exponential':
            weighted_best = self.custom_sort(best_list)
            self.rank_weights = list(range(7,0,-1))
            exp_wts = []
            for i in range(len(self.rank_weights)):
                tmp = self.rank_weights[i]*self.decay**abs((self.rank_weights[i] - self.rank_weights[0]))
                exp_wts.append(tmp)
            exp_np = np.array(exp_wts)
            exp_wts = exp_np/np.sum(exp_np)
            best_param = np.average(weighted_best, weights=np.flip(exp_wts, axis=0), axis=0)
        return best_param


    def assemble_optimizers(self):
        '''initialize all optimizers 1. DE, 2. POS, 3. APSO, 4.FF, 5. CuckooSearch, 6.flower pollinatiion, 7.Bat Algorithm'''

        self.DE = Differential_Evolution(self.obj, self.class_obj,num_agents=self.num_agents, num_gen=self.num_gen,scaling=0.9
                                         ,seed=self.seed, cross_prob=0.50, target=0.0001,obj_dim=self.obj_dim)

        self.PSO = PSO(self.obj, self.class_obj,num_agents= self.num_agents, num_gen=self.num_gen, seed= self.seed,
                       obj_dim = self.obj_dim, alpha=2,alpha_decay=0.03,beta = 2,floor = self.class_obj.floor,
                       ceiling=self.class_obj.ceiling)

        self.PSOL = PSOLevy(self.obj, self.class_obj,num_agents= self.num_agents, num_gen=self.num_gen, seed= self.seed,
                        obj_dim = self.obj_dim, alpha=2,alpha_decay=0.03,beta = 2,floor = self.class_obj.floor,
                        ceiling=self.class_obj.ceiling)

        self.Cuckoo = CuckooSearch(self.obj, self.class_obj,num_agents= self.num_agents, num_gen=self.num_gen, seed= self.seed,
                                   obj_dim = self.obj_dim, alpha=0.01, p=0.25,beta = 1.5,floor = self.class_obj.floor,
                                   ceiling=self.class_obj.ceiling)

        self.flower =  FlowerPollination(self.obj, self.class_obj,num_agents= self.num_agents, num_gen=self.num_gen,
                                         seed= self.seed, obj_dim = self.obj_dim, gamma=0.1, p=0.5,beta=1.5, floor= self.class_obj.floor,
                                         ceiling = self.class_obj.ceiling)
        self.Bat = BAT(self.obj, self.class_obj,num_agents= self.num_agents, num_gen=self.num_gen,
                                     seed= self.seed, obj_dim = self.obj_dim, fmin=0.01, fmax=0.25, pulse=0.2, amplitude=1,
                                     scaling=0.01, floor= self.class_obj.floor, ceiling = self.class_obj.ceiling)
        self.BatL = BATLevy(self.obj, self.class_obj,num_agents= self.num_agents, num_gen=self.num_gen,
                       seed= self.seed, obj_dim = self.obj_dim, fmin=0.01, fmax=0.25, pulse=0.2, amplitude=1,
                       scaling=0.01, floor= self.class_obj.floor, ceiling = self.class_obj.ceiling)


    def model_fit(self):
        self.assemble_optimizers()
        for n in range(self.num_gen):
            de_best = self.DE.ensemble_fit()
            pso_best = self.PSO.ensemble_fit()
            psol_best = self.PSOL.ensemble_fit()
            bat_best = self.Bat.ensemble_fit()
            batl_best = self.BatL.ensemble_fit()
            cuckoo_best = self.Cuckoo.ensemble_fit()
            flower_best = self.flower.ensemble_fit()

            best_list = np.array([de_best, pso_best, psol_best, bat_best, batl_best,cuckoo_best, flower_best])
            if self.scheme!='meta':
                self.ensemble_best = self.weighing_scheme(best_list)
            else:
                self.ensemble_best = self.meta_weighing_scheme(best_list)


            '''now communicate this ensemble best result to all the optimizers'''
            '''
                1. For DE update the current best with the ensemble best
                2. For PSO update the current best with the ensemble best
                3. For apso update the current best with the ensemble best
                4. For Firefly update the worst with the ensemble best
                5. For Cuckoo update the worst in tmp nest with the ensemble best
            '''
            if n%self.comms==0:
                #print('communicating')
                self.DE.current_best = self.ensemble_best
                self.PSO.current_best = self.ensemble_best
                self.PSOL.current_best = self.ensemble_best
                self.Bat.current_best = self.ensemble_best
                self.BatL.current_best = self.ensemble_best
                #randint(0, 9)
                self.Cuckoo.population[random.randint(0, self.num_agents -1)] = self.ensemble_best
                self.flower.population[random.randint(0, self.num_agents -1)] = self.ensemble_best

            self.final_best = np.array([de_best, pso_best, psol_best, bat_best, batl_best,cuckoo_best, flower_best])
            self.final_best_params = self.custom_sort(best_list)[len(best_list) - 1]
            self.err_tracker.append(abs(self.class_obj.return_value(self.final_best_params)  - self.class_obj.min))
        #self.current_best = self.custom_sort(self.population)[len(self.population) - 1]
        self.final_best_params = self.custom_sort(best_list)[len(best_list) - 1]
        return self.final_best_params

    def best_param(self):
        return self.custom_sort(self.final_best)[len(self.final_best) - 1]



def load_iris_data():
    file='iris.csv'
    data_df = pd.read_csv(file)
    df_1 = data_df.sample(frac=1)
    new_df = df_1['species']
    new_df[new_df=='setosa']=0
    new_df[new_df=='versicolor']=1
    new_df[new_df=='virginica']=2
    data_ary = np.asarray(df_1).astype(object)
    ip_labels, op_Lables = np.asarray(data_ary[:, :data_ary.shape[1]-1]).astype(float), np.asarray(new_df).astype(int)
    ip_norm_lable = (ip_labels - ip_labels.mean(axis=0)) / ip_labels.std(axis=0)
    ip_labels = np.hstack([ip_labels, np.ones((ip_labels.shape[0], 1))])
    return ip_labels, op_Lables


def load_dataset_2():
    '''Train, Validation, Test Set'''
    file='imageseg.csv'
    data_df = pd.read_csv(file)
    df_1 = data_df.sample(frac=1)

    new_df = df_1['CLASS']
    '''brickface, sky, foliage, cement, window, path, grass'''
    new_df[new_df=='BRICKFACE'], new_df[new_df=='SKY'], new_df[new_df=='FOLIAGE'] = 0, 1, 2
    new_df[new_df=='CEMENT'], new_df[new_df=='WINDOW'], new_df[new_df=='PATH'], new_df[new_df=='GRASS']= 3, 4, 5, 6



    data_ary = np.asarray(df_1).astype(object)
    ip_labels, op_Lables = np.asarray(data_ary[:, 1:data_ary.shape[1]]).astype(float), np.asarray(new_df).astype(int)

    '''train, test, validation splits'''
    t_s, v_s, s_s = 0.6, 0.2, 0.2
    train_end = int(round(len(ip_labels)*t_s))
    valid_end = train_end + int(round(len(ip_labels)*v_s))


    ip_train, op_train = ip_labels[0:train_end, :], op_Lables[0:train_end]
    ip_valid, op_valid = ip_labels[train_end:valid_end, :], op_Lables[train_end:valid_end]
    ip_test, op_test = ip_labels[valid_end:, :], op_Lables[valid_end: ]

    return ip_train, op_train, ip_valid, op_valid, ip_test, op_test


if __name__ == '__main__':



    '''basic testing of PSO Optimizer
    t0 = time.time()
    procedure()
    print time.time() - t0,    
    '''
    X,y, X_v, y_v, X_t, t_t = load_dataset_2()
    dim = X.shape[1]
    num_classes = np.max(y) + 1
    params = 0.001*np.random.randn(dim, num_classes)
    obj = loss(params, X, y)
    param_new = np.reshape(params, params.shape[0]*params.shape[1])
    toggle_params = np.reshape(param_new,(dim, num_classes))
    model = EnsembleOptimizer(obj.entropy_loss, obj, num_agents=100, num_gen= 1000, seed=7,obj_dim=param_new.shape[0], communicate=10,scheme='best', wts_decay=0.2, floor=-50, ceiling=50)
    #model = PSO(obj.entropy_loss, obj,num_agents=100, num_gen=500,seed=10,obj_dim=param_new.shape[0])
    #model = CuckooSearch(obj.four_peak_func, obj,num_agents=50, num_gen=100, seed=100,obj_dim=params.shape[0])
    t0 = time.time()
    best_param = model.model_fit()
    best_param = model.best_param()
    #best_param = model.best_param()
    print('EO time taken is: ')
    print(time.time() - t0)
    toggle_params = np.reshape(best_param,(dim, num_classes))
    value = obj.return_value(best_param)
    print(value)
    accuracy  = obj.predict(toggle_params, X, y)
    print('train accuracy is:  ')
    print(accuracy)
    print('valid accuracy is: ')
    val_accuracy  = obj.predict(toggle_params, X_v, y_v)
    print(val_accuracy)





