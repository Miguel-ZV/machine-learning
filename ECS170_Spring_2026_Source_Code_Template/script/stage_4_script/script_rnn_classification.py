from local_code.stage_4_code.Dataset_Loader_Classification import Dataset_Loader
from local_code.stage_4_code.Method_RNN_Classifcation import Method_RNN_Classifcation
from local_code.stage_4_code.Result_Saver import Result_Saver
from local_code.stage_4_code.Evaluate_Metrics import Evaluate_Metrics
import numpy as np
import torch

from local_code.stage_4_code.Setting_Train_Test_Predfined import Setting_Train_Test_Predefined

# ---- Multi-Layer Perceptron script ----
if 1:
    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    # ------------------------------------------------------

    # ---- objection initialization section ---------------
    data_obj = Dataset_Loader('predefined test and train data', '')
    data_obj.dataset_source_folder_path = '../../data/stage_4_data/text_classification/'

    method_obj = Method_CNN_CIFAR('recurrent neural network for text classification', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_4_result/CNN_CIFAR_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Train_Test_Predefined('predefined train test sets', '')

    evaluate_obj = Evaluate_Metrics('metrics', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    metric_scores = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('RNN_Classification Accuracy: ' + str(metric_scores['accuracy']))
    print('RNN_Classification Precision: ' + str(metric_scores['precision']))
    print('RNN_Classification Recall: ' + str(metric_scores['recall']))
    print('RNN_Classification F1: ' + str(metric_scores['f1']))
    print('************ Finish ************')
    # ------------------------------------------------------


