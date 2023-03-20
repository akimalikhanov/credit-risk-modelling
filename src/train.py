import sys
sys.path.insert(1, '../app')
from utils import *
import config
import time


start_time=time.time()

def full_train():
    path_dict=config.PATH_DICT
    params=config.PARAMS
    print('TRAIN PIPELINE')
    timer()
    # print('\n*** 0. Saving Original Dtypes ***')
    # save_dtypes(list(config.DTYPES_DICT.keys()), config.PATH_DICT, config.DTYPES_DICT)
    # timer()
    print('\n*** 1. Feature Generation ***')
    train=full_df(path_dict, 
                mode='train')
    timer()
    print('\n*** 2. Correlation ***')
    train=correlation_filter(train, 
                            thresh=0.9, 
                            corr_path=path_dict['corr_matrix'], 
                            mode='train')
    timer()
    print('\n*** 3. Missing Variables ***')
    train=missing_filter(train, 
                        thresh=80, 
                        col_path=path_dict['missing_columns_drop'], 
                        mode='train')
    timer()
    print('\n*** 4. Zero Variance ***')
    train=zero_var_filter(train, 
                        col_path=path_dict['zero_variance_drop'], 
                        mode='train')
    timer()
    print('\n*** 5. Zero Importance ***')
    train=drop_zero_imp(train, 
                        feat_imp_path=path_dict['zero_imp_drop'], 
                        k=5, 
                        params=params, 
                        mode='train', 
                        drop_by='importance (gain)')
    timer()
    save_data(train, path_dict['train_ready_file'], path_dict['dtypes'])
    print('\n*** 6. Model Training ***')
    train_model(train, 
                params=params, 
                model_path=path_dict['model_file'],
                col_tran_path=path_dict['lgb_ohe'])
    
    print(f'\n*** DONE : {(time.time()-start_time)/60:.3f} Minutes ***')


if __name__ == "__main__":
    full_train()
