
'''
This file contains the implementation of Flower Pollination Optimizer
'''
import time
import numpy as np
import math as m
import random
from objective_func import *

class init_FP:

    def __init__(self, obj, obj_dim,num_fireflies, ceiling, floor, seed, **kwargs):
        self.obj_dim = obj_dim
        self.objective = obj
        self.num_fireflies = num_fireflies
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

class FlowerPollination:

    def __init__(self, obj, class_obj,num_agents, num_gen, seed, obj_dim, gamma, p=0.25,beta=1.5, floor=-5, ceiling = 5,**kwargs):
        self.obj = obj
        self.num_agents = num_agents
        self.class_obj = class_obj
        self.population = None
        self.num_gen = num_gen
        self.seed = seed
        self.obj_dim = obj_dim
        self.gamma = gamma
        self.p = p
        self.beta = beta
        self.randomize = None
        self.runner = None
        self.floor = floor
        self.ceiling = ceiling
        self.fitness = None
        self.current_best=None
        self.current_best_particle=None
        self.param_bound = [self.floor, self.ceiling]
        self.levy = None
        self.sigma_v, self.sigma_u = None, None
        self.err_tracker = []
        self.init_ensemble()

    def new_intensity(self, param):
        return self.obj(param)

    def init_optimization(self):
        self.runner = init_FP(self.obj, self.obj_dim, self.num_agents, self.ceiling, self.floor, self.seed)
        self.population=self.runner.init_pop
        self.fitness = self.runner.fitness

    def scale_params(self, param_list):
        for i in range(len(param_list)):
            if param_list[i]>self.param_bound[1]:
                param_list[i] = self.param_bound[1]
            elif param_list[i]<self.param_bound[0]:
                param_list[i] = self.param_bound[0]

        return param_list


    def custom_sort(self, arr):
        l = list(arr)
        l.sort(key=lambda crunch: self.new_intensity(crunch))
        sort_ary = np.asarray(l)
        return sort_ary


    def Calc_Levy_Flights(self):
        self.sigma_u = (m.gamma(1+self.beta)*m.sin(m.pi*self.beta/2)/(m.gamma((1+self.beta)/2)*self.beta*2**((self.beta-1)/2)))**(1/self.beta)
        self.sigma_v = 1

        #for i in range(len(self.population)):
        U = np.random.normal(0, self.sigma_u, size=self.obj_dim)
        V = np.random.normal(0, 1, size=self.obj_dim)
        s = U/(abs(V)**(1/self.beta))
            #self.population[i]+=self.gamma*s*(self.population[i]-self.current_best)*np.random.randn(len(self.population[i]))
            #self.population[i]=self.scale_params(self.population[i])
        self.levy = s

    def model_fit(self):
        self.init_optimization()
        self.current_best = self.custom_sort(self.population)[len(self.population) - 1]
        for n in range(self.num_gen):
            for i in range(len(self.population)):
                tmp_pop_save = np.copy(self.population[i])
                if np.random.uniform(0, 1) > self.p:
                    self.Calc_Levy_Flights()
                    self.population[i]+=self.gamma*self.levy*(self.current_best - self.population[i])
                    #self.population[i]+=self.gamma*self.levy*(self.population[i] - self.current_best)
                else:
                    [k1, k2] = random.sample(list(self.population), 2)
                    x_diff_1 = [k1_i - k2_i for k1_i, k2_i in zip(k1, k2)]
                    add_val = np.array([random.uniform(0, 1)*x_diff_i for x_diff_i in x_diff_1])
                    #self.population[i]+=np.random.uniform(0, 1)*x_diff_1
                    self.population[i]+=add_val
                if self.new_intensity(tmp_pop_save) >= self.new_intensity(self.population[i]):
                    self.population[i] = tmp_pop_save
            self.current_best = self.custom_sort(self.population)[len(self.population) - 1]
            self.err_tracker.append(abs(self.class_obj.return_value(self.current_best)  - self.class_obj.min))
        return self.current_best


    def init_ensemble(self):
        self.init_optimization()
        self.current_best = self.custom_sort(self.population)[len(self.population) - 1]

    def ensemble_fit(self):
        for i in range(len(self.population)):
            tmp_pop_save = np.copy(self.population[i])
            if np.random.uniform(0, 1) > self.p:
                self.Calc_Levy_Flights()
                self.population[i]+=self.gamma*self.levy*(self.current_best - self.population[i])
                #self.population[i]+=self.gamma*self.levy*(self.population[i] - self.current_best)
            else:
                [k1, k2] = random.sample(list(self.population), 2)
                x_diff_1 = [k1_i - k2_i for k1_i, k2_i in zip(k1, k2)]
                add_val = np.array([random.uniform(0, 1)*x_diff_i for x_diff_i in x_diff_1])
                #self.population[i]+=np.random.uniform(0, 1)*x_diff_1
                self.population[i]+=add_val
            if self.new_intensity(tmp_pop_save) >= self.new_intensity(self.population[i]):
                self.population[i] = tmp_pop_save
        self.current_best = self.custom_sort(self.population)[len(self.population) - 1]
        self.err_tracker.append(abs(self.class_obj.return_value(self.current_best)  - self.class_obj.min))
        return self.current_best



if __name__ == '__main__':

    params = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    tmp_1 = params.shape
    obj = rosenbrock(params)
    tmp = obj.rosenbrock_func
    t0 = time.time()

    model = FlowerPollination(obj.rosenbrock_func, obj,num_agents=100, num_gen=2000, seed=7, obj_dim=params.shape[0], gamma=0.1, p=0.5,beta=1.5, floor=-5, ceiling = 5)
    best_param = model.model_fit()
    print('FA time taken is: ')
    print(time.time() - t0)

    value = obj.return_value(best_param)
    global_val_1 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    print('true global value is: ')
    print(obj.return_value(global_val_1))
    #global_val_2 = tmp([0, -4])
    print('best param value is')
    print(value)
    print(best_param)
    print('end here')









