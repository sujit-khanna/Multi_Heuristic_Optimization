
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

NUM_TRIALS = 10

def load_dataset_1():
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
    t_s, v_s, s_s = 0.6, 0.2, 0.2
    train_end = int(round(len(ip_labels)*t_s))
    valid_end = train_end + int(round(len(ip_labels)*v_s))


    ip_train, op_train = ip_labels[0:train_end, :], op_Lables[0:train_end]
    ip_valid, op_valid = ip_labels[train_end:valid_end, :], op_Lables[train_end:valid_end]
    ip_test, op_test = ip_labels[valid_end:, :], op_Lables[valid_end: ]

    return ip_train, op_train, ip_valid, op_valid, ip_test, op_test



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


def load_dataset_3():
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
    '''train, test, validation splits'''
    t_s, v_s, s_s = 0.6, 0.2, 0.2
    train_end = int(round(len(ip_labels)*t_s))
    valid_end = train_end + int(round(len(ip_labels)*v_s))


    ip_train, op_train = ip_labels[0:train_end, :], op_Lables[0:train_end]
    ip_valid, op_valid = ip_labels[train_end:valid_end, :], op_Lables[train_end:valid_end]
    ip_test, op_test = ip_labels[valid_end:, :], op_Lables[valid_end: ]

    return ip_train, op_train, ip_valid, op_valid, ip_test, op_test

def test_svm_sgd():
    ''' reg param = [0.001, 0.01, 0.1, 1]
        l_r = [0.001, 0.01, 0.1, 1]
    '''
    reg_params = [0, 0.001, 0.01, 0.1, 1]
    learning_rate = [0.001, 0.01, 0.1, 1]
    num_iter = 1000
    num_trials = NUM_TRIALS


    data_sets = [load_dataset_1(), load_dataset_2()]
    set_names = ['wisconsin', 'image_seg']

    c= -1
    log_list = []
    for data in data_sets:
        c+=1
        ip_train, op_train, ip_valid, op_valid, ip_test, op_test = data
        for l_r in learning_rate:
            for reg in reg_params:
                tmp_log = []
                svm_ver = str(set_names[c]) + '_l_r_' + str(l_r) + '_reg_lambda_' + str(reg)
                tmp_log.append(svm_ver)
                mean_loss, mean_train_acc, mean_valid_acc = [], [], []
                loss_err, stdev_train_acc, stdev_valid_acc = [], [], []
                for j in range(num_trials):
                    svm = LinearSVM()
                    loss_hist = svm.train(ip_train, op_train, learning_rate=l_r, reg=reg, num_iters=num_iter, verbose=False)
                    op_train_pred = svm.predict(ip_train)
                    train_acc = np.mean(op_train == op_train_pred)
                    valid_pred = svm.predict(ip_valid)
                    valid_acc = np.mean(op_valid == valid_pred)
                    mean_loss.append(loss_hist[len(loss_hist) - 1]), mean_train_acc.append(train_acc), mean_valid_acc.append(valid_acc)


                tmp_log.append(statistics.mean(mean_loss)), tmp_log.append(statistics.mean(mean_train_acc)), tmp_log.append(statistics.mean(mean_valid_acc))
                tmp_log.append(statistics.stdev(mean_loss)/num_trials), tmp_log.append(statistics.stdev(mean_train_acc)), tmp_log.append(statistics.stdev(mean_valid_acc))
                log_list.append(tmp_log)


    log_arr = np.array(log_list)
    return log_arr


def test_svm_eo():
    '''
    reg param = [0.001, 0.01, 0.1, 1]
    comms = [all five techniques]
    comms freq = [1, 10, 100]
    :return:
    '''
    comm_type = ['average', 'rank', 'exponential', 'best', 'meta']
    comms_freq = [1, 10, 100]
    reg_params = [0]
    num_trials = NUM_TRIALS
    num_iter = 1000


    data_sets = [load_dataset_1(), load_dataset_2()]
    set_names = ['wisconsin', 'image_seg']


    c= -1
    log_list = []
    for data in data_sets:
        c+=1
        ip_train, op_train, ip_valid, op_valid, ip_test, op_test = data
        for comms in comm_type:
            for freq in comms_freq:
                for reg in reg_params:
                    tmp_log = []
                    svm_ver = str(set_names[c]) + '_CommType_' + str(comms) + '_Freq_' + str(freq) + '_reg_lambda_' + str(reg)
                    tmp_log.append(svm_ver)
                    mean_loss, mean_train_acc, mean_valid_acc = [], [], []
                    for j in range(num_trials):
                        set_seed = random.randint(1, 1000)
                        dim = ip_train.shape[1]
                        num_classes = np.max(op_train) + 1
                        params = 0.001*np.random.randn(dim, num_classes)
                        obj = loss(params, ip_train, op_train)
                        param_new = np.reshape(params, params.shape[0]*params.shape[1])
                        toggle_params = np.reshape(param_new,(dim, num_classes))
                        model = EnsembleOptimizer(obj.entropy_loss, obj, num_agents=100, num_gen= num_iter, seed=set_seed,obj_dim=param_new.shape[0], communicate=freq,
                                                  scheme=comms, wts_decay=0.2, floor=-10, ceiling=10)
                        best_param = model.model_fit()
                        best_param = model.best_param()
                        loss_val = obj.return_value(best_param)
                        toggle_best_params = np.reshape(best_param,(dim, num_classes))
                        train_acc  = obj.predict(toggle_best_params, ip_train, op_train)
                        valid_acc = obj.predict(toggle_best_params, ip_valid, op_valid)
                        print('loss: ' + str(loss_val) + ' train acc: ' + str(train_acc) + ' valid acc: ' + str(valid_acc))
                        mean_loss.append(loss_val), mean_train_acc.append(train_acc), mean_valid_acc.append(valid_acc)

                    tmp_log.append(statistics.mean(mean_loss)), tmp_log.append(statistics.mean(mean_train_acc)), tmp_log.append(statistics.mean(mean_valid_acc))
                    tmp_log.append(statistics.stdev(mean_loss)/num_trials), tmp_log.append(statistics.stdev(mean_train_acc)), tmp_log.append(statistics.stdev(mean_valid_acc))
                    log_list.append(tmp_log)


    log_arr = np.array(log_list)
    return log_arr




def make_plots():
    pass



if __name__ == '__main__':
    sgd_results_arr = test_svm_sgd()
    sgd_results_df = pd.DataFrame(sgd_results_arr)
    sgd_results_df.to_csv('SVM_SGD_results.csv')

    mmo_results_arr = test_svm_eo()
    mmo_results_df = pd.DataFrame(mmo_results_arr)
    mmo_results_df.to_csv('SVM_MMO_results.csv')
