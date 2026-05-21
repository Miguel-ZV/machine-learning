'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from local_code.stage_3_code.Evaluate_Metrics import Evaluate_Metrics
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

from ECS170_Spring_2026_Source_Code_Template.local_code.stage_3_code.script_data_loader import instance


class Method_RNN_Classification(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 200
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3

    instance_length = None

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        (method.__init__
         (self, mName, mDescription))
        nn.Module.__init__(self)
        self.rnn = nn.RNN(50 , 100, 1)
        self.fc_layer = nn.Linear(100, 2)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        '''Forward propagation'''
        # rnn
        output, h_n = self.rnn(x)
        # output layer result
        y_pred = self.fc_layer(h_n)
        return y_pred

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train(self, X, y):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-2)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        # for training accuracy investigation purpose
        evaluator = Evaluate_Metrics('training evaluator', '')

        loss_values = []

        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself
        for epoch in range(self.max_epoch): # you can do an early stop if self.max_epoch is too much...
            # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
            y_pred = self.forward(torch.FloatTensor(np.array(X)))
            # convert y to torch.tensor as well
            y_true = torch.LongTensor(np.array(y)) - 1
            # calculate the training loss
            train_loss = loss_function(y_pred, y_true)

            # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
            optimizer.zero_grad()
            # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
            # do the error backpropagation to calculate the gradients
            train_loss.backward()
            # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
            # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
            optimizer.step()

            evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
            scores = evaluator.evaluate()
            loss = train_loss.item()
            loss_values.append(loss)

            if epoch%100 == 0:
                print('Epoch:', epoch, 'Accuracy:', scores['accuracy'], 'Precision:', scores['precision'],
                      'Recall:', scores['recall'], 'F1:', scores['f1'], 'Loss:', loss)

        plt.plot(range(self.max_epoch), loss_values)
        plt.title('ORL')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()
    
    def test(self, X):
        # do the testing, and result the result
        y_pred = self.forward(torch.FloatTensor(np.array(X)))
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return y_pred.max(1)[1]
    
    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X']) + 1
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}