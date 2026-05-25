'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset
import numpy as np
import string
from nltk.tokenize import word_tokenize


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    data_instance_length = None

    def clean(self, text):
        # Lowercase and tokenize, but KEEP punctuation and stopwords.
        # NLTK's word_tokenize handles punctuation well (e.g., separates "?" from words).
        tokens = word_tokenize(text.lower())
        return tokens

    def create_vocab(self, all_tokens):
        # Create mapping from word -> integer ID, and ID -> word
        unique_tokens = set(all_tokens)
        self.vocab = {word: i for i, word in enumerate(unique_tokens)}
        self.inverse_vocab = {i: word for i, word in enumerate(unique_tokens)}
        self.vocab_size = len(self.vocab)

    def load(self):
        print('loading and processing data...')

        # 1. Read all jokes and tokenize
        all_tokens = []
        with open(self.dataset_source_folder_path + self.dataset_source_file_name, 'r', encoding='utf-8') as f:
            for line in f:
                split_line = line.split(",")
                joke = " ".join(split_line[1:])  # Assuming CSV format: id,joke_text
                tokens = self.clean(joke)

                # Append an end-of-joke token so the model knows when to stop generating
                tokens.append("[EOS]")
                all_tokens.extend(tokens)

        # 2. Build vocabulary
        self.create_vocab(all_tokens)

        # 3. Convert all text to integer IDs
        encodings = [self.vocab[word] for word in all_tokens]

        # 4. Create sliding windows (X = 3 words, Y = 1 word)
        X_train = []
        y_train = []

        for i in range(len(encodings) - self.data_instance_length):
            X_train.append(encodings[i: i + self.data_instance_length])
            y_train.append(encodings[i + self.data_instance_length])

        return {'X_train': X_train, 'y_train': y_train, 'vocab': self.vocab, 'inverse_vocab': self.inverse_vocab}