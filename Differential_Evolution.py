
'''
This file contains the implementation of differential evolution optimizer
'''
import time
import numpy as np
import math as m
import random
from objective_func import *

class init_DE:

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


class Differential_Evolution:

    def __init__(self, obj, class_obj,num_agents, num_gen, seed, scaling, cross_prob, obj_dim,floor = -5,celiling=5, **kwargs):
        self.obj = obj
        self.class_obj = class_obj
        self.num_agents = num_agents
        self.population = None
        self.num_gen = num_gen
        self.seed = seed
        self.obj_dim = obj_dim
        self.alpha = 0.2
        self.alpha_decay = 0.03
        self.F = scaling
        self.Cr = cross_prob
        self.randomize = None
        self.runner = None
        self.floor = floor
        self.ceiling = celiling
        self.fitness = None
        self.current_best=None
        self.velocity_bound = [self.floor, self.ceiling]
        self.velocity=0
        self.err_tracker = []
        self.init_ensemble()


    def new_intensity(self, param):
        return self.obj(param)

    def init_optimization(self):
        self.runner = init_DE(self.obj, self.obj_dim, self.num_agents, self.ceiling, self.floor, self.seed)
        self.population=self.runner.init_pop
        self.fitness = self.runner.fitness

    def run_optimization(self):
        return None

    def custom_sort(self, arr):
        l = list(arr)
        l.sort(key=lambda crunch: self.new_intensity(crunch))
        sort_ary = np.asarray(l)
        return sort_ary

    def scale_velocity(self):
        for i in range(len(self.velocity)):
            if self.velocity[i]>self.velocity_bound[1]:
                self.velocity[i] = self.velocity_bound[1]
            elif self.velocity[i]<self.velocity_bound[0]:
                self.velocity[i] = self.velocity_bound[0]

    def model_fit(self):
        self.init_optimization()
        self.current_best = self.custom_sort(self.population)[len(self.population) - 1]
        for n in range(self.num_gen):
            storage_vector =np.asarray(np.zeros(self.population.shape)).astype(float)
            for i in range(len(self.population)):
                tmp_var = list(self.population)
                [k1, k2, k3, k4] = random.sample(list(self.population), 4)
                x_diff_1 = [k1_i - k2_i for k1_i, k2_i in zip(k1, k2)]
                x_diff_2 = [k3_i - k4_i for k3_i, k4_i in zip(k3, k4)]

                self.velocity = np.array([best_i + self.F*(x_diff_1_i + x_diff_2_i) for best_i, x_diff_1_i, x_diff_2_i in zip(self.current_best, x_diff_1, x_diff_2)])
                self.scale_velocity()

                cr_thresh = random.uniform(0, 1)
                idx_thresh = random.randint(0, self.obj_dim+1)
                for j in range(self.obj_dim):
                    if cr_thresh<=self.Cr or idx_thresh==j:
                        storage_vector[i] = self.velocity
                    else:
                        storage_vector[i] = self.population[i]

            for k in range(len(self.population)):
                if self.new_intensity(storage_vector[k]) >= self.new_intensity(self.population[k]):
                    self.population[k] = storage_vector[k]

            self.current_best = self.custom_sort(self.population)[len(self.population) - 1]
            self.err_tracker.append(abs(self.class_obj.return_value(self.current_best)  - self.class_obj.min))

        return self.current_best

    def init_ensemble(self):
        self.init_optimization()
        self.current_best = self.custom_sort(self.population)[len(self.population) - 1]

    def ensemble_fit(self):
        storage_vector =np.asarray(np.zeros(self.population.shape)).astype(float)
        for i in range(len(self.population)):
            tmp_var = list(self.population)
            [k1, k2, k3, k4] = random.sample(list(self.population), 4)
            x_diff_1 = [k1_i - k2_i for k1_i, k2_i in zip(k1, k2)]
            x_diff_2 = [k3_i - k4_i for k3_i, k4_i in zip(k3, k4)]

            self.velocity = np.array([best_i + self.F*(x_diff_1_i + x_diff_2_i) for best_i, x_diff_1_i, x_diff_2_i in zip(self.current_best, x_diff_1, x_diff_2)])
            self.scale_velocity()

            cr_thresh = random.uniform(0, 1)
            idx_thresh = random.randint(0, self.obj_dim+1)
            for j in range(self.obj_dim):
                if cr_thresh<=self.Cr or idx_thresh==j:
                    storage_vector[i] = self.velocity
                else:
                    storage_vector[i] = self.population[i]

        for k in range(len(self.population)):
            if self.new_intensity(storage_vector[k]) >= self.new_intensity(self.population[k]):
                self.population[k] = storage_vector[k]

        self.current_best = self.custom_sort(self.population)[len(self.population) - 1]
        self.err_tracker.append(abs(self.class_obj.return_value(self.current_best)  - self.class_obj.min))

        return self.current_best



if __name__ == '__main__':
    '''basic testing of Differential Evolution'''
    params = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    tmp_1 = params.shape
    obj = rosenbrock(params)
    tmp = obj.rosenbrock_func
    t0 = time.time()
    model = Differential_Evolution(obj.rosenbrock_func, obj,num_agents=100, num_gen=2000,scaling=0.9 ,seed=7, cross_prob=0.50, target=0.0001,obj_dim=params.shape[0])
    best_param = model.model_fit()
    print('DE time taken is: ')
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


