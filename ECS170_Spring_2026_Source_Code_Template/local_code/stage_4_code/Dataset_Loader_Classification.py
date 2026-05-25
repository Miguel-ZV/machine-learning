'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset
import string
from pathlib import Path
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    data_instance_length = None
    glove_source_folder_path = None
    glove_file_name = None
    stop_words = set(stopwords.words('english'))
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load_glove(self):
        glove = {}

        f = open(self.glove_source_folder_path + self.glove_file_name, 'r', encoding='utf-8')
        for line in f:
            split_line = line.split()
            word = " ".join(split_line[:-50])
            embedding = np.array([float(val) for val in split_line[-50:]])
            glove[word] = embedding
        f.close()

        return glove

    def clean(self, text, glove):
        tokens = word_tokenize(text)
        tokens = [w.lower() for w in tokens]
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        words = [word for word in stripped if word.isalpha()]
        words = [w for w in words if not w in self.stop_words]
        embeddings = [glove[word] if word in glove else [0]*50 for word in words]

        if len(embeddings) < self.data_instance_length:
            for i in range(self.data_instance_length - len(embeddings)):
                embeddings.append([0]*50)
        elif len(embeddings) > self.data_instance_length:
            embeddings = embeddings[:self.data_instance_length]

        return embeddings

    def load(self):
        print('loading data...')

        X_train = []
        y_train = []
        X_test = []
        y_test = []

        glove = self.load_glove()

        # negative training reviews
        p = Path(self.dataset_source_folder_path + "/train/neg/")
        for instance in p.iterdir():
            f = open(instance, 'r', encoding='utf-8')
            text = f.read()
            X_train.append(self.clean(text,glove))
            y_train.append(0)
            f.close()

        # positive training reviews
        p = Path(self.dataset_source_folder_path + "/train/pos/")
        for instance in p.iterdir():
            f = open(instance, 'r', encoding='utf-8')
            text = f.read()
            X_train.append(self.clean(text, glove))
            y_train.append(1)
            f.close()

        # negative testing reviews
        p = Path(self.dataset_source_folder_path + "/test/neg/")
        for instance in p.iterdir():
            f = open(instance, 'r', encoding='utf-8')
            text = f.read()
            X_test.append(self.clean(text, glove))
            y_test.append(0)
            f.close()

        # positive testing reviews
        p = Path(self.dataset_source_folder_path + "/test/pos/")
        for instance in p.iterdir():
            f = open(instance, 'r', encoding='utf-8')
            text = f.read()
            X_test.append(self.clean(text, glove))
            y_test.append(1)
            f.close()

        return {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}