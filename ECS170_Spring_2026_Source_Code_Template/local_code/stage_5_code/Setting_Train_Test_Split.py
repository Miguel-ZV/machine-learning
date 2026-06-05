'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.setting import setting

class Setting_Train_Test_Split(setting):
    
    def load_run_save_evaluate(self):
        
        # load dataset
        loaded_data = self.dataset.load()

        train_indices = loaded_data['train_test_val']['idx_train']
        test_indices = loaded_data['train_test_val']['idx_test']
        val_indices = loaded_data['train_test_val']['idx_val']
        # run MethodModule
        self.method.data = {'train': train_indices, 'test': test_indices, 'val': val_indices, 'graph': loaded_data['graph']}
        learned_result = self.method.run()
            
        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()
            
        self.evaluate.data = learned_result
        
        return self.evaluate.evaluate()