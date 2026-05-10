from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN_CIFAR import Method_CNN_CIFAR
from local_code.stage_3_code.Result_Saver import Result_Saver
from local_code.stage_3_code.Evaluate_Metrics import Evaluate_Metrics
import numpy as np
import torch

from local_code.stage_3_code.Setting_Train_Test_Predfined import Setting_Train_Test_Predefined

# ---- Multi-Layer Perceptron script ----
if 1:
    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    # ------------------------------------------------------

    # ---- objection initialization section ---------------
    data_obj = Dataset_Loader('predefined test and train data', '')
    data_obj.dataset_source_folder_path = '../../data/stage_3_data/'
    data_obj.dataset_source_file_name = 'CIFAR'

    method_obj = Method_CNN_CIFAR('convention neural network for CIFAR dataset', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_3_result/CNN_CIFAR_'
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
    print('CIFAR Accuracy: ' + str(metric_scores['accuracy']))
    print('CIFAR Precision: ' + str(metric_scores['precision']))
    print('CIFAR Recall: ' + str(metric_scores['recall']))
    print('CIFAR F1: ' + str(metric_scores['f1']))
    print('************ Finish ************')
    # ------------------------------------------------------


