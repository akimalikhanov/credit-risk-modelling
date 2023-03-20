PATH_DICT={
    'application_train': '../data/application_train.csv',
    'application_test': '../data/application_test.csv',
    'bur': '../data/bureau.csv',
    'bb': '../data/bureau_balance.csv',
    'previous': '../data/previous_application.csv',
    'cash': '../data/POS_CASH_balance.csv',
    'installments': '../data/installments_payments.csv',
    'card_balance': '../data/credit_card_balance.csv',
    'ohe_dict': '../models/pipeline/ohe_dict.pkl',
    'corr_matrix': '../data/corr_matrix.csv',
    'missing_columns_drop': '../models/pipeline/missing_columns_drop.pkl',
    'zero_variance_drop': '../models/pipeline/zero_var_columns_drop.pkl',
    'zero_imp_drop': '../models/pipeline/zero_imp.pkl',
    'model_file': '../models/model.txt',
    'lgb_ohe': '../models/pipeline/lgb_ohe.pkl',
    'train_ready_file': '../data/train_ready.csv',
    'test_ready_file': '../data/test_ready.csv',
    'dtypes': '../models/pipeline/dtypes_final.pkl',
    'submit': '../data/submit.csv',
    'bur_temp': '../data/templates/bur_temp.pkl',
    'bb_temp': '../data/templates/bb_temp.pkl',
    'prev_temp': '../data/templates/prev_temp.pkl',
    'cash_temp': '../data/templates/cash_temp.pkl',
    'inst_temp': '../data/templates/inst_temp.pkl',
    'card_temp': '../data/templates/card_temp.pkl',
    'agg_dict': '../models/pipeline/agg_dict.pkl'
    }

DF_NAMES=['application_test', 'bur', 'bb', 'previous', 'cash', 'installments', 'card_balance', 'ohe_dict']

SAMPLE_DTYPES_DICT='../models/pipeline/sample_dtypes_dict.pkl'

# DTYPES_DICT={
#     'application_train': '../models/pipeline/dtypes/app_dtypes.pkl',
#     'bur': '../models/pipeline/dtypes/bureau_dtypes.pkl',
#     'bb': '../models/pipeline/dtypes/bb_dtypes.pkl',
#     'previous': '../models/pipeline/dtypes/previous_dtypes.pkl',
#     'cash': '../models/pipeline/dtypes/cash_dtypes.pkl',
#     'installments': '../models/pipeline/dtypes/inst_dtypes.pkl',
#     'card_balance': '../models/pipeline/dtypes/card_dtypes.pkl',
# }

PARAMS={
    'num_boost_round': 10000,
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.05,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'subsample': 0.8,
    'n_jobs': -1,
    'random_state': 5,
    'verbose': -1
    }

CLIENTS_FILE='../data/clients/'

CLIENT_FILENAMES=['application.csv', 
                    'bureau.csv', 
                    'bureau_balance.csv', 
                    'previous_app.csv', 
                    'cash.csv', 
                    'installments.csv', 
                    'card.csv']