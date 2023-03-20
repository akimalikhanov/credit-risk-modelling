import sys
sys.path.insert(1, '../app')
from utils import *
import config
import time


start_time=time.time()


def full_test():
    path_dict=config.PATH_DICT
    print('TEST PIPELINE')
    timer()
    print('\n*** 1. Feature Generation ***')
    test=full_df(path_dict, 
                mode='test')
    timer()
    print('\n*** 2. Correlation ***')
    test=correlation_filter(test, 
                            thresh=0.9, 
                            corr_path=path_dict['corr_matrix'], 
                            mode='test')
    timer()
    print('\n*** 3. Missing Variables ***')
    test=missing_filter(test, 
                        thresh=80, 
                        col_path=path_dict['missing_columns_drop'], 
                        mode='test')
    timer()
    print('\n*** 4. Zero 3 ***')
    test=zero_var_filter(test, 
                        col_path=path_dict['zero_variance_drop'], 
                        mode='test')
    timer()
    print('\n*** 5. Zero Importance ***')
    test=drop_zero_imp(test, 
                        feat_imp_path=path_dict['zero_imp_drop'], 
                        k=5, 
                        params=None, 
                        mode='test', 
                        drop_by='importance (gain)')
    timer()
    save_data(test, path_dict['test_ready_file'])
    print('\n*** 6. Making Prediction ***')
    make_prediction(test, 
                    model_path=path_dict['model_file'],
                    col_tran_path=path_dict['lgb_ohe'],
                    save_path=path_dict['submit'])
        
    print(f'\n*** DONE : {(time.time()-start_time)/60:.2f} Minutes ***')


if __name__ == "__main__":
    full_test()