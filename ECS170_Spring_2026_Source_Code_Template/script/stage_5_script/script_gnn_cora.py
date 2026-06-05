from local_code.stage_5_code.Dataset_Loader import Dataset_Loader
from local_code.stage_5_code.Method_GNN_Cora import Method_GNN_Cora
from local_code.stage_5_code.Result_Saver import Result_Saver
from local_code.stage_5_code.Evaluate_Metrics import Evaluate_Metrics
import numpy as np
import torch

from local_code.stage_5_code.Setting_Train_Test_Split import Setting_Train_Test_Split

# ---- Multi-Layer Perceptron script ----
if 1:
    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    # ------------------------------------------------------

    # ---- objection initialization section ---------------
    data_obj = Dataset_Loader('split test and train data', '')
    data_obj.dataset_source_folder_path = '../../data/stage_5_data/cora'
    data_obj.dataset_name= 'cora'

    method_obj = Method_GNN_Cora('graph neural network for pubmed dataset', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_5_result/GNN_PUBMED_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Train_Test_Split('split train test sets', '')

    evaluate_obj = Evaluate_Metrics('metrics', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    metric_scores = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('CORA Accuracy: ' + str(metric_scores['accuracy']))
    print('CORA Precision: ' + str(metric_scores['precision']))
    print('CORA Recall: ' + str(metric_scores['recall']))
    print('CORA F1: ' + str(metric_scores['f1']))
    print('************ Finish ************')
    # ------------------------------------------------------


