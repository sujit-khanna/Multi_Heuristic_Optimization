
'''
This file contains the implementation of the BatAlgorithm Algorithm
'''
import time
import numpy as np
import math as m
import random
from objective_func import *


class init_BAT:

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

class BAT:

    def __init__(self, obj, class_obj,num_agents, num_gen, seed, obj_dim, fmin=0.01, fmax=0.25, beta=1.5, pulse=0.1,amplitude=1,scaling=0.01,gamma=0.9,floor=-5, ceiling = 5,**kwargs):
        self.obj = obj
        self.num_agents = num_agents
        self.class_obj = class_obj
        self.population = None
        self.num_gen = num_gen
        self.seed = seed
        self.obj_dim = obj_dim
        self.fmin = fmin
        self.fmax = fmax
        self.f = None
        self.beta = beta
        self.pulse = pulse
        self.r = [self.pulse for i in range(num_agents)]
        self.Amp = amplitude
        self.A =[self.Amp for i in range(num_agents)]
        self.randomize = None
        self.runner = None
        self.floor = floor
        self.ceiling = ceiling
        self.fitness = None
        self.current_best=None
        self.gamma=gamma
        self.current_best_particle=None
        self.velocities = None
        self.param_bound = [self.floor, self.ceiling]
        self.levy = None
        self.sigma_v, self.sigma_u = None, None
        self.err_tracker = []
        self.scaling = scaling
        self.mean_amp = None
        self.init_ensemble()


    def new_intensity(self, param):
        return self.obj(param)


    def init_optimization(self):
        self.runner = init_BAT(self.obj, self.obj_dim, self.num_agents, self.ceiling, self.floor, self.seed)
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


    def model_fit(self):
        self.init_optimization()
        self.current_best = self.custom_sort(self.population)[len(self.population) - 1]
        self.current_best_particle=self.population
        self.velocities =np.asarray(np.zeros(self.population.shape)).astype(float)

        for n in range(self.num_gen):
            self.current_best = self.custom_sort(self.population)[len(self.population) - 1]
            population_copy = np.copy(self.population)
            self.mean_amp = np.mean(self.A)
            for i in range(len(self.population)):
                self.f = self.fmin + (self.fmax - self.fmin)*random.uniform(0,1)
                tmp_best = np.copy(self.current_best_particle[i])
                self.velocities[i]+=(self.population[i] - self.current_best)*self.f
                population_copy[i] += self.velocities[i]
                population_copy[i] = self.scale_params(population_copy[i])

                if random.random()>self.r[i]:
                    population_copy[i] = self.current_best + self.scaling*np.random.uniform(-1.0,1.0)*self.mean_amp
                    population_copy[i] = self.scale_params(population_copy[i])

                if self.new_intensity(population_copy[i]) > self.new_intensity(self.population[i]) and np.random.uniform(0,1)<self.A[i]:
                    self.population[i]=np.copy(population_copy[i])
                if self.new_intensity(self.population[i]) > self.new_intensity(self.current_best):
                    self.current_best = np.copy(self.population[i])
                    self.A[i] = self.Amp*self.A[i]
                    self.r[i] = self.pulse*(1-np.exp(-1*self.gamma*i))
            self.current_best = self.custom_sort(self.population)[len(self.population) - 1]
            self.err_tracker.append(abs(self.class_obj.return_value(self.current_best)  - self.class_obj.min))
        return self.current_best

    def init_ensemble(self):
        self.init_optimization()
        self.current_best = self.custom_sort(self.population)[len(self.population) - 1]
        self.current_best_particle=self.population
        self.velocities =np.asarray(np.zeros(self.population.shape)).astype(float)


    def ensemble_fit(self):
        population_copy = np.copy(self.population)
        self.mean_amp = np.mean(self.A)
        for i in range(len(self.population)):
            self.f = self.fmin + (self.fmax - self.fmin)*random.uniform(0,1)
            tmp_best = np.copy(self.current_best_particle[i])
            self.velocities[i]+=(self.population[i] - self.current_best)*self.f
            population_copy[i] += self.velocities[i]
            population_copy[i] = self.scale_params(population_copy[i])

            if random.random()>self.r[i]:
                population_copy[i] = self.current_best + self.scaling*np.random.uniform(-1.0,1.0)*self.mean_amp
                population_copy[i] = self.scale_params(population_copy[i])

            if self.new_intensity(population_copy[i]) > self.new_intensity(self.population[i]) and np.random.uniform(0,1)<self.A[i]:
                self.population[i]=np.copy(population_copy[i])
            if self.new_intensity(self.population[i]) > self.new_intensity(self.current_best):
                self.current_best = np.copy(self.population[i])
                self.A[i] = self.Amp*self.A[i]
                self.r[i] = self.pulse*(1-np.exp(-1*self.gamma*i))
        self.current_best = self.custom_sort(self.population)[len(self.population) - 1]
        self.err_tracker.append(abs(self.class_obj.return_value(self.current_best)  - self.class_obj.min))
        return self.current_best





if __name__ == '__main__':

    '''basic testing of PSO Optimizer
    t0 = time.time()
    procedure()
    print time.time() - t0,    
    '''

    params = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    tmp_1 = params.shape
    obj = rosenbrock(params)
    tmp = obj.rosenbrock_func
    #(self, obj, class_obj,num_agents, num_gen, seed, obj_dim, fmin=0.01, fmax=0.25, beta=1.5, pulse=0.2,amplitude=1,scaling=0.01,gamma=0.9,floor=-5, ceiling = 5,**kwargs):
    model = BAT(obj.rosenbrock_func, obj,num_agents=100, num_gen=5000,seed=10,obj_dim=params.shape[0])
    #model = CuckooSearch(obj.four_peak_func, obj,num_agents=50, num_gen=100, seed=100,obj_dim=params.shape[0])
    t0 = time.time()
    best_param = model.model_fit()
    #best_param = model.best_param()
    print('PSO time taken is: ')
    print(time.time() - t0)
    value = obj.return_value(best_param)

    global_val_1 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    print('true global value is: ')
    print(obj.return_value(global_val_1))
    #global_val_2 = tmp([0, -4])
    print('best param value is')
    print(value)
    print(best_param)
    print('end here')







