import math
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statistics
from utilities.classifiers.linear_classifier import LinearSVM
from utilities.classifiers.linear_svm import svm_loss_naive
from utilities.classifiers.linear_svm import svm_loss_vectorized
from EO import *


NUM_GEN=1000
NUM_TRIALS = 10


def load_dataset_2_old():
    '''Train, Validation, Test Set'''
    file='wisconsin.csv'
    data_df = pd.read_csv(file)
    df_1 = data_df.sample(frac=1)
    df_1[df_1['nuclei']=='?']=5

    new_df = df_1['class']
    '''2-benign, 4-malignant'''
    new_df[new_df==2]=0
    new_df[new_df==4]=1
    new_df[new_df==5]=1
    data_ary = np.asarray(df_1).astype(object)
    ip_labels, op_Lables = np.asarray(data_ary[:, 1:data_ary.shape[1]-1]).astype(float), np.asarray(new_df).astype(int)

    '''train, test, validation splits'''
    t_s, v_s = 0.8, 0.2
    train_end = int(round(len(ip_labels)*t_s))
    valid_end = train_end + int(round(len(ip_labels)*v_s))


    ip_train, op_train = ip_labels[0:train_end, :], op_Lables[0:train_end]
    ip_test, op_test = ip_labels[train_end:, :], op_Lables[train_end: ]

    return ip_train, op_train, ip_test, op_test



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
    t_s, v_s = 0.8, 0.2
    train_end = int(round(len(ip_labels)*t_s))
    valid_end = train_end + int(round(len(ip_labels)*v_s))


    ip_train, op_train = ip_labels[0:train_end, :], op_Lables[0:train_end]
    ip_test, op_test = ip_labels[train_end:, :], op_Lables[train_end: ]

    return ip_train, op_train, ip_test, op_test


#NUM_GEN=5
#NUM_TRIALS = 3
def list_EO():
    print('Ensemble Optimizer')
    pop_size, num_gen, num_trials = 100, NUM_GEN, NUM_TRIALS
    comms, scheme, decay = 1, 'exponential', 0.2

    ip_train, op_train, ip_test, op_test = load_dataset_2()
    train_err = []
    mean_loss, mean_train_acc, mean_test_acc, tmp_log = [], [], [], []
    for i in range(num_trials):
        set_seed = random.randint(1, 1000)
        dim = ip_train.shape[1]
        num_classes = np.max(op_train) + 1
        params = 0.001*np.random.randn(dim, num_classes)
        obj = loss(params, ip_train, op_train)
        param_new = np.reshape(params, params.shape[0]*params.shape[1])
        toggle_params = np.reshape(param_new,(dim, num_classes))
        model = EnsembleOptimizer(obj.entropy_loss, obj, num_agents=100, num_gen= num_gen, seed=set_seed,obj_dim=param_new.shape[0], communicate=comms,
                                  scheme=scheme, wts_decay=0.2, floor=-50, ceiling=50)
        best_param = model.model_fit()
        best_param = model.best_param()
        loss_val = obj.return_value(best_param)
        toggle_best_params = np.reshape(best_param,(dim, num_classes))
        train_acc  = obj.predict(toggle_best_params, ip_train, op_train)
        test_acc = obj.predict(toggle_best_params, ip_test, op_test)
        mean_loss.append(loss_val), mean_train_acc.append(train_acc), mean_test_acc.append(test_acc)

        print('MMO train acc: ' + str(train_acc) + ' test acc: ' + str(test_acc))
        train_err.append(model.err_tracker)

    df = pd.DataFrame(train_err)
    df_ary = np.asarray(df).astype(float)
    mean_ary, var_ary = np.mean(df_ary, axis=0), np.std(df_ary, axis=0)
    tmp_log.append(statistics.mean(mean_loss)), tmp_log.append(statistics.mean(mean_train_acc)), tmp_log.append(statistics.mean(mean_test_acc))

    print('Mean Loss is: ' + str(tmp_log[0]) + ' Mean Train Accuracy is: ' + str(tmp_log[1]) + ' Mean Test Accuracy is: ' + str(tmp_log[2]))

    return mean_ary, var_ary


def list_SGD():
    print('SGD')
    pop_size, num_gen, num_trials = 100, NUM_GEN, NUM_TRIALS
    l_r, l2_reg, decay = 0.01, 0, 0.2

    ip_train, op_train, ip_test, op_test = load_dataset_2()
    train_err = []
    mean_loss, mean_train_acc, mean_test_acc, tmp_log = [], [], [], []

    for i in range(num_trials):
        svm = LinearSVM()
        loss_hist = svm.train(ip_train, op_train, learning_rate=l_r, reg=l2_reg, num_iters=num_gen, verbose=False)
        op_train_pred = svm.predict(ip_train)
        train_acc = np.mean(op_train == op_train_pred)
        test_pred = svm.predict(ip_test)
        test_acc = np.mean(op_test == test_pred)
        train_err.append(loss_hist)
        print('SGD train acc: ' + str(train_acc) + ' test acc: ' + str(test_acc))
        mean_loss.append(loss_hist[len(loss_hist) - 1]), mean_train_acc.append(train_acc), mean_test_acc.append(test_acc)
    df = pd.DataFrame(train_err)
    df_ary = np.asarray(df).astype(float)
    mean_ary, var_ary = np.mean(df_ary, axis=0), np.std(df_ary, axis=0)
    tmp_log.append(statistics.mean(mean_loss)), tmp_log.append(statistics.mean(mean_train_acc)), tmp_log.append(statistics.mean(mean_test_acc))


    print('Mean Loss is: ' + str(tmp_log[0]) + ' Mean Train Accuracy is: ' + str(tmp_log[1]) + ' Mean Test Accuracy is: ' + str(tmp_log[2]))

    return mean_ary, var_ary



if __name__ == '__main__':

    '''This creates SGD Plot '''

    mean, var = list_SGD()
    x = np.linspace(0, NUM_GEN, NUM_GEN)
    fig, ax = plt.subplots()
    plt.plot(x, mean, color='#1B2ACC', label='MMO')
    ax.set(xlabel='Iterations', ylabel='loss', title='Training Loss')
    plt.legend()
    plt.fill_between(x, mean - var, mean + var, alpha=0.5, edgecolor='#1B2ACC', facecolor='#089FFF')
    plt.savefig('SGD' +'_train_loss' +'.png')
    plt.close()


    '''This creates MMO Plot'''

    mean, var = list_EO()
    x = np.linspace(0, NUM_GEN, NUM_GEN)
    fig, ax = plt.subplots()
    plt.plot(x, mean, color='#1B2ACC', label='MMO')
    ax.set(xlabel='Iterations', ylabel='loss', title='Training Loss')
    plt.legend()
    plt.fill_between(x, mean - var, mean + var, alpha=0.5, edgecolor='#1B2ACC', facecolor='#089FFF')
    plt.savefig('MMO' +'_train_loss' +'.png')
    plt.close()


