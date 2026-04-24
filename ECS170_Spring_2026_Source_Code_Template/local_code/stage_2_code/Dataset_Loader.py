'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    train_dataset_source_file_name = None
    test_dataset_source_file_name = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load(self):
        print('loading data...')
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        f1 = open(self.dataset_source_folder_path + self.train_dataset_source_file_name, 'r')
        f2 = open(self.dataset_source_folder_path + self.test_dataset_source_file_name, 'r')
        for line in f1:
            line = line.strip('\n')
            elements = [int(i) for i in line.split(',')]
            X_train.append(elements[1:])
            y_train.append(elements[0])
        for line in f2:
            line = line.strip('\n')
            elements = [int(i) for i in line.split(',')]
            X_test.append(elements[1:])
            y_test.append(elements[0])
        f1.close()
        f2.close()
        return {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}