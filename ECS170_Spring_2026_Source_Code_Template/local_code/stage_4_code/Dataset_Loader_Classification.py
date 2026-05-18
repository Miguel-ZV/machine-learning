'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset
import string
from pathlib import Path
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def clean(text):
    tokens = word_tokenize(text)
    tokens = [w.lower() for w in tokens]
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()]
    words = [w for w in words if not w in set(stopwords.words('english'))]
    return words


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading data...')

        X_train = []
        y_train = []
        X_test = []
        y_test = []

        # negative training reviews
        p = Path(self.dataset_source_folder_path + "/train/neg/")
        for instance in p.iterdir():
            f = open(instance, 'r')
            text = f.read()
            X_train.append(clean(text))
            y_train.append(0)
            f.close()

        # positive training reviews
        p = Path(self.dataset_source_folder_path + "/train/pos/")
        for instance in p.iterdir():
            f = open(instance, 'r')
            text = f.read()
            X_train.append(clean(text))
            y_train.append(0)
            f.close()

        # negative testing reviews
        p = Path(self.dataset_source_folder_path + "/test/neg/")
        for instance in p.iterdir():
            f = open(instance, 'r')
            text = f.read()
            X_train.append(clean(text))
            y_train.append(0)
            f.close()

        # positive testing reviews
        p = Path(self.dataset_source_folder_path + "/test/pos/")
        for instance in p.iterdir():
            f = open(instance, 'r')
            text = f.read()
            X_train.append(clean(text))
            y_train.append(0)
            f.close()

        return {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}