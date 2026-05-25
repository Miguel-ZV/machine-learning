'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.setting import setting

class Setting_Train_Generation(setting):

    X_train = None
    y_train = None

    def load_run(self):

        # load dataset
        loaded_data = self.dataset.load()

        X_train = loaded_data['X_train']
        y_train= loaded_data['y_train']
        vocab = loaded_data['vocab']
        inverse_vocab = loaded_data['inverse_vocab']
        vocab_size = len(vocab)

        # run MethodModule
        self.method.data = {'train': {'X': X_train, 'y': y_train}}
        self.method.data = {'train': {'X': X_train, 'y': y_train}}
        self.method.vocab = vocab
        self.method.inverse_vocab = inverse_vocab
        self.method.vocab_size = vocab_size
        self.method.run()

        return