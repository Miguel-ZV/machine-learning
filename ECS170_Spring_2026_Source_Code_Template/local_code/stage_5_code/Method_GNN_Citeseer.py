'''
Concrete MethodModule class for a specific learning MethodModule
'''
from torch.cuda import graph

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
#from local_code.stage_5_code.Evaluate_Metrics import Evaluate_Metrics
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from local_code.stage_5_code.layers import GraphConvolution
import copy


class Method_GNN_Citeseer(method, nn.Module):
    data = None
    max_epoch = 600
    learning_rate = 0.01*2

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.gc1 = GraphConvolution(3703, 36)
        self.gc2 = GraphConvolution(36, 6)
        self.dropout = 0.5

    def forward(self, x, adj):

        x = self.gc1(x, adj)
        x = nn.functional.relu(x)
        x = nn.functional.dropout(x, training=self.training)
        x = self.gc2(x, adj)

        return nn.functional.log_softmax(x, dim=1)

    def train(self, X, y):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=5e-4)
        loss_values = []

        best_val_loss = float('inf')
        best_weights = None
        patience = 20
        bad_counter = 0

        super().train()  # Set to training mode

        for epoch in range(self.max_epoch):
            optimizer.zero_grad()

            output = self.forward(X, self.data['graph']['utility']['A'])

            loss_train = nn.functional.nll_loss(output[self.data['train']], y[self.data['train']])
            loss_train.backward()
            optimizer.step()

            loss_values.append(loss_train.item())

            super().train(False)
            with torch.no_grad():
                output_val = self.forward(X, self.data['graph']['utility']['A'])
                loss_val = nn.functional.nll_loss(output_val[self.data['val']], y[self.data['val']])

            if loss_val < best_val_loss:
                best_val_loss = loss_val
                best_weights = copy.deepcopy(self.state_dict())
                bad_counter = 0
            else:
                bad_counter += 1

            if bad_counter == patience:
                break

            if epoch % 10 == 0:
                print(f'Epoch: {epoch:04d}, Loss: {loss_train.item():.4f}')

        plt.plot(range(len(loss_values) - bad_counter), loss_values[:len(loss_values) - bad_counter])
        plt.title('GNN Citeseer')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()
        self.load_state_dict(best_weights)

    def test(self, X):
        super().train(False)
        # do the testing, and result the result
        y_pred = self.forward(torch.FloatTensor(np.array(X)), self.data['graph']['utility']['A'])
        y_pred = y_pred[self.data['test']]
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return y_pred.max(1)[1]

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['graph']['X'], self.data['graph']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['graph']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['graph']['y'][self.data['test']]}