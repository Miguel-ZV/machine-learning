from local_code.stage_4_code.Dataset_Loader_Generation import Dataset_Loader
from local_code.stage_4_code.Method_RNN_Generation import Method_RNN_Generation
from local_code.stage_4_code.Result_Saver import Result_Saver
from local_code.stage_4_code.Evaluate_Metrics import Evaluate_Metrics
import numpy as np
import torch

from local_code.stage_4_code.Setting_Train_Generation import Setting_Train_Generation

# ---- Multi-Layer Perceptron script ----
if 1:
    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    instance_length = 20
    # ------------------------------------------------------

    # ---- objection initialization section ---------------
    data_obj = Dataset_Loader('predefined test and train data', '')
    data_obj.dataset_source_folder_path = '../../data/stage_4_data/text_generation/'
    data_obj.dataset_source_file_name = 'data'
    data_obj.glove_source_folder_path = '../../data/stage_4_data/'
    data_obj.glove_file_name = 'wiki_giga_2024_50_MFT20_vectors_seed_123_alpha_0.75_eta_0.075_combined.txt'
    data_obj.data_instance_length = instance_length

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    method_obj = Method_RNN_Generation('recurrent neural network for text generation', '')
    method_obj.to(device)
    method_obj.instance_length = instance_length

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_4_result/RNN_Generation_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Train_Generation('training for generation model', '')

    evaluate_obj = Evaluate_Metrics('metrics', '')

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    setting_obj.load_run()


