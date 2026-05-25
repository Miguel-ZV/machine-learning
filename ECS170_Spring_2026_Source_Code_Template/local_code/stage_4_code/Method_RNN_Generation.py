'''
Concrete MethodModule class for a specific learning MethodModule
'''
from sympy import true

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from local_code.stage_4_code.Evaluate_Metrics import Evaluate_Metrics
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


class Method_RNN_Generation(method, nn.Module):

    max_epoch = 30
    learning_rate = 1e-3
    hidden_size = 128
    instance_length = None
    vocab = None
    inverse_vocab = None
    vocab_size = None

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

    def init_model(self):
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=50)
        self.input_dropout = nn.Dropout(p=0.2)

        self.rnn = nn.LSTM(
            input_size=50,
            hidden_size=self.hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.5
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, self.vocab_size)
        )

    def forward(self, x):
        '''Forward propagation'''
        embedded = self.embedding(x)
        embedded = self.input_dropout(embedded)

        output, (h_n, c_n) = self.rnn(embedded)

        last_step_output = output[:, -1, :]

        y_pred = self.fc_layer(last_step_output)

        return y_pred

    def train(self, X, y):
        super().train()

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-2)

        dataset = torch.utils.data.TensorDataset(torch.LongTensor(X), torch.LongTensor(y))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=2e-3,
            steps_per_epoch=len(dataloader),
            epochs=self.max_epoch
        )

        loss_function = nn.CrossEntropyLoss(label_smoothing=0.05)

        loss_values = []

        device = next(self.parameters()).device

        for epoch in range(self.max_epoch):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                y_pred = self.forward(batch_X)
                y_true = batch_y

                train_loss = loss_function(y_pred, y_true)

                optimizer.zero_grad()
                train_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

                optimizer.step()

                scheduler.step()

                loss = train_loss.item()
                epoch_loss += loss

            avg_epoch_loss = epoch_loss / len(dataloader)
            loss_values.append(avg_epoch_loss)

            if epoch % 2 == 0 or epoch == self.max_epoch - 1:
                print(f'Epoch: {epoch:02d} | Loss: {avg_epoch_loss:.4f}')

        plt.plot(range(self.max_epoch), loss_values)
        plt.title('RNN Generation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

    def generate(self, seed_sequence, max_words=30):
        super().train(False)

        device = next(self.parameters()).device

        encoded_seed = [self.vocab[word] for word in seed_sequence]

        current_sequence = encoded_seed.copy()
        generated_words = [self.inverse_vocab[idx] for idx in current_sequence]

        with torch.no_grad():
            for _ in range(max_words):
                x_input = torch.LongTensor([current_sequence[-self.instance_length:]]).to(device)

                y_pred = self.forward(x_input)[0]

                probabilities = F.softmax(y_pred, dim=0).cpu().numpy()
                next_word_id = np.random.choice(len(probabilities), p=probabilities)
                next_word = self.inverse_vocab[next_word_id]

                if next_word == "[EOS]":  # Matched to Dataset_Loader token
                    break

                generated_words.append(next_word)
                current_sequence.append(next_word_id)

        return " ".join(generated_words)

    def run(self):
        print('method running...')
        self.init_model()
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start generating...')
        while (True):
            print(self.generate(input("Enter three words:").lower().split()))
        return