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


class Method_CNN_CIFAR(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 500
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3*1.1

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        (method.__init__
         (self, mName, mDescription))
        nn.Module.__init__(self)
        self.conv1 = nn.Conv2d(3, 24, 5, padding='same')
        self.conv2 = nn.Conv2d(24, 64, 5, padding='same')
        self.conv3 = nn.Conv2d(64, 128, 5, padding='same')
        self.fc_layer_1 = nn.Linear(2048, 256)
        self.fc_layer_2 = nn.Linear(256, 64)
        self.fc_layer_3 = nn.Linear(64, 32)
        self.fc_layer_4 = nn.Linear(32, 10)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        '''Forward propagation'''
        #reorder
        x = x.permute(0, 3, 1, 2)
        # conv layers and pools
        x = nn.functional.leaky_relu(self.conv1(x))
        x = nn.functional.max_pool2d(x,(2,2))
        x = nn.functional.leaky_relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, (2, 2))
        x = nn.functional.leaky_relu(self.conv3(x))
        x = nn.functional.max_pool2d(x, (2, 2))
        # flatten
        x = torch.flatten(x, 1)
        # fc layers
        x = nn.functional.leaky_relu(self.fc_layer_1(x))
        x = nn.functional.leaky_relu(self.fc_layer_2(x))
        x = nn.functional.leaky_relu(self.fc_layer_3(x))
        # output layer result
        y_pred = self.fc_layer_4(x)
        return y_pred

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train(self, X, y):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-2*1.2)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        # for training accuracy investigation purpose
        evaluator = Evaluate_Metrics('training evaluator', '')

        #mini-batch
        dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True)

        loss_values = []

        # it will be an iterative gradient updating process
        for epoch in range(self.max_epoch): # you can do an early stop if self.max_epoch is too much...
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
                y_pred = self.forward(batch_X)
                # convert y to torch.tensor as well
                y_true = batch_y
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
                epoch_loss += loss

            loss_values.append(epoch_loss/500)

            if epoch%10== 0:
                print('Epoch:', epoch)

        plt.plot(range(self.max_epoch), loss_values)
        plt.title('CIFAR')
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
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}