import numpy as np
import h5py
import re
from keras.utils.data_utils import Sequence
from keras.utils.np_utils import to_categorical


def shuffle_aligned_list_sar(data):
    num = len(data)
    p = np.random.permutation(num)
    return [data[p[d]] for d in range(num)]


def h5read(filename, datasets):
    if type(datasets) != list:
        datasets = [datasets]
    with h5py.File(filename, 'r') as h5f:
        data = [h5f[ds][:] for ds in datasets]
    return data

def read_file_paths(file_name, shuffle=False):
    with open(file_name) as f:
        list_batch = []
        for line in f:
            list_batch.append(line)

    if shuffle == True:
        list_batch = shuffle_aligned_list_sar(list_batch)

    return list_batch


def read_data_train(dir_path, batchX, num_class, time_step, input_dim, label_size):
    feature_sets = np.zeros((len(batchX), time_step, input_dim))
    lab = []
    for line in range(len(batchX)):
        str_0 = batchX[line].rstrip()
        str_1= str_0.split(' ')[0]

        name = dir_path
        temp = str_1.split('_')
        lab.append(temp[2])
        feature_sets[line, :, :] = h5read(name,str_1)[0]

    lab = np.array(lab, dtype=int) - 1

    lab = np.tile(lab, [label_size, 1])
    lab = lab.T
    lab = lab.ravel()
    labels = np.eye(num_class)[lab]

    del lab
    return feature_sets, labels



class GenSequence_AVG(Sequence):
    def __init__(self, data, dir_path, batch_size, num_class, time_step, input_dim, label_size, shuffle):
        self.data = data
        self.dir_path = dir_path
        self.batch_size = batch_size
        self.num_class = num_class
        self.time_step = time_step
        self.input_dim = input_dim
        self.label_size = label_size
        self.shuffle = shuffle

    def __len__(self):
        return len(self.data) // self.batch_size

    def __getitem__(self,idx):

        batchX = self.data[idx * self.batch_size:(idx + 1) * self.batch_size]
        trainX, trainY = read_data_train(self.dir_path, batchX, num_class=self.num_class, time_step=self.time_step, input_dim=self.input_dim,
                                       label_size=self.label_size)

        return trainX, trainY

    def on_epoch_end(self):
        if self.shuffle:
            self.data = shuffle_aligned_list_sar(self.data)



###################################################################

def read_test_data(dir_path, list_test, time_step,input_dim):
    feature_sets = np.zeros((len(list_test), time_step,input_dim))
    for line in range(len(list_test)):
        str_1 = list_test[line].rstrip()
        str_1 = re.split("[	\' ]+", str_1)[0]
        load_name = dir_path
        feature_sets[line, :, :] = (h5read(load_name,str_1)[0])

    return feature_sets


def read_test_data_long(dir_path, list_test, input_dim):

    str_1 = list_test.rstrip()
    str_1 = re.split("[	\' ]+", str_1)[0]
    load_name = dir_path
    feature_sets = h5read(load_name,str_1)[0]
    feature_sets = np.reshape(feature_sets, (-1, feature_sets.shape[0], input_dim))

    return feature_sets

def read_test_data_long_pred_gen(dir_path, list_test, input_dim):

    str_1 = list_test[0].rstrip()
    str_1 = re.split("[ ]+", str_1)[0]
    load_name = dir_path
    feature_sets = h5read(load_name,str_1)[0]
    feature_sets = np.reshape(feature_sets, (-1, feature_sets.shape[0], input_dim))

    return feature_sets


class GenSequence_Test(Sequence):
    def __init__(self, data, dir_path, batch_size, time_step, input_dim):
        self.data = data
        self.dir_path = dir_path
        self.batch_size = batch_size
        self.time_step = time_step
        self.input_dim = input_dim


    def __len__(self):
        return len(self.data) // self.batch_size

    def __getitem__(self,idx):

        batchX = self.data[idx * self.batch_size:(idx + 1) * self.batch_size]
        testX= read_test_data(self.dir_path, batchX, self.time_step,self.input_dim)

        return testX


class GenSequence_Test_long(Sequence):
    def __init__(self, data, dir_path, batch_size, input_dim):
        self.data = data
        self.dir_path = dir_path
        self.batch_size = batch_size
        self.input_dim = input_dim


    def __len__(self):
        return len(self.data) // self.batch_size

    def __getitem__(self,idx):

        batchX = self.data[idx * self.batch_size:(idx + 1) * self.batch_size]
        testX = read_test_data_long_pred_gen(self.dir_path, batchX, self.input_dim)

        return testX

