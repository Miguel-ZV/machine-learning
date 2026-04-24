'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score



class Evaluate_Metrics(evaluate):
    data = None
    
    def evaluate(self):
        #print('evaluating performance...')
        return {'accuracy': accuracy_score(self.data['true_y'],self.data['pred_y']),
                'precision': precision_score(self.data['true_y'], self.data['pred_y'], average='macro', zero_division=0.0),
                'recall': recall_score(self.data['true_y'], self.data['pred_y'], average='macro'),
                'f1': f1_score(self.data['true_y'], self.data['pred_y'], average='macro')}
        