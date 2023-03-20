import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import normaltest
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import sys
import gc



def plot_numeric(df, cols, num_plots):
    count=1
    for _ in range(0, int(np.ceil(len(cols)/num_plots))):
        fig, axes = plt.subplots(1, num_plots, figsize=(18, 3))
        for ax in axes.flat:
            if cols:
                col_name=cols.pop(0)
                ax.hist(df[col_name])
                _, p=normaltest(df[col_name], nan_policy='omit')
                test_res='Non-Normal' if p<0.05 else 'Normal'
                ax.set_title(f'{count}--{col_name}\n({test_res})')
                fig.tight_layout()
                count+=1


def plot_categorical(df, cols, num_rows, r=0):
    count=1
    for _ in range(0, int(np.ceil(len(cols)/num_rows))):
        fig, axes = plt.subplots(1, num_rows, figsize=(21, 5))
        for ax in axes.flat:
            if cols:
                col_name=cols.pop(0)
                vals=df[col_name].value_counts()
                x, y=vals.index, vals.values
                ax.bar(x, y)
                ax.set_title(f'{count}--{col_name}')
                for tick in ax.get_xticklabels():
                    tick.set_rotation(r)
                ax.bar_label(ax.containers[0])
                count+=1


def custom_lgbm_cv(features, target, col_tran, k=5):
    metric_df=pd.DataFrame(columns=['Train AUC', 'Test AUC'])
    feat_importances=[]
    kfold=StratifiedKFold(k)
    for f, (tr, te) in enumerate(kfold.split(features, y=target)):
        X_train, y_train=features.iloc[tr, :], target.iloc[tr]
        X_test, y_test=features.iloc[te, :], target.iloc[te]

        X_train_tr=col_tran.fit_transform(X_train)
        X_test_tr=col_tran.transform(X_test)
        weight=np.count_nonzero(y_train==0)/np.count_nonzero(y_train==1)

        params={'num_boost_round': 10000,
                'objective': 'binary',
                'scale_pos_weight': weight,
                'metric': 'auc',
                'learning_rate': 0.05,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'subsample': 0.8,
                'n_jobs': -1,
                'random_state': 5,
                'verbose': -1}

        dtrain=lgb.Dataset(X_train_tr, label=y_train)
        dval=lgb.Dataset(X_test_tr, label=y_test)

        model=lgb.train(
                params=params,
                train_set=dtrain,
                valid_sets=[dtrain, dval],
                valid_names=['train', 'test'],
                callbacks=[lgb.early_stopping(100, verbose=-1), lgb.log_evaluation(-1)])
        
        test_score, train_score=model.best_score['test']['auc'], model.best_score['train']['auc']
        metric_df.loc[f]=[train_score, test_score]

        feat_importances.append(model.feature_importance(importance_type='gain'))
    
    feat_importances=np.array(feat_importances).mean(axis=0)
    feat_importances_df=pd.DataFrame({'feature': col_tran.get_feature_names_out(),
                                        'importance': feat_importances})
    metric_df.loc['Avg']=[metric_df['Train AUC'].mean(), metric_df['Test AUC'].mean()]
    return metric_df, feat_importances_df


def miss_table(data):
    miss_table=data.isna().sum().to_frame(name='Count')
    miss_table['Percent']=miss_table['Count']/len(data)*100
    miss_table['Dtype']=data.dtypes[miss_table.index]
    miss_table['Count']=miss_table['Count'].replace({0: np.nan})
    miss_table=miss_table.dropna()
    print(f"There are {len(miss_table)}/{data.shape[1]} columns with missing values")
    print('Distribution by dtypes:')
    print(miss_table['Dtype'].value_counts())
    return miss_table.sort_values(by='Count', ascending=False)



def return_size(df):
    """Return size of dataframe in gigabytes"""
    return round(sys.getsizeof(df) / 1e9, 2)

def convert_types(df, print_info = False):
    
    original_memory = df.memory_usage().sum()
    
    # Iterate through each column
    for c in df:
        
        # Convert ids and booleans to integers
        if ('SK_ID' in c):
            df[c] = df[c].fillna(0).astype(np.int32)
            
        # Convert objects to category
        # elif (df[c].dtype == 'object') and (df[c].nunique() < df.shape[0]):
        #     df[c] = df[c].astype('category')
        
        # Booleans mapped to integers
        elif list(df[c].unique()) == [1, 0]:
            df[c] = df[c].astype(bool)
        
        # Float64 to float32
        elif df[c].dtype == float:
            df[c] = df[c].astype(np.float32)
            
        # Int64 to int32
        elif df[c].dtype == int:
            df[c] = df[c].astype(np.int32)
        
    new_memory = df.memory_usage().sum()
    
    if print_info:
        print(f'Original Memory Usage: {round(original_memory / 1e9, 2)} gb.')
        print(f'New Memory Usage: {round(new_memory / 1e9, 2)} gb.')
        
    return df


def generate_domain_features(df):
    print(f'Before adding features: {df.shape}')

    # CREDIT related 
    bins=[18, 35, 40, 50, 60, 70, 120]
    labels=['18-29', '30-39', '40-49', '50-59', '60-69', '70+']
    df['NEW_AGE_GROUP']=pd.cut(df['DAYS_BIRTH']/-365, bins=bins, labels=labels, right=False)

    cred_by_contract=df.groupby('NAME_CONTRACT_TYPE')['AMT_CREDIT'].mean() 
    cred_by_housing_type=df.groupby('NAME_HOUSING_TYPE')['AMT_CREDIT'].mean() 
    cred_by_org_type=df.groupby('ORGANIZATION_TYPE')['AMT_CREDIT'].mean() 
    cred_by_education_type=df.groupby('NAME_EDUCATION_TYPE')['AMT_CREDIT'].mean() 
    cred_by_gender=df.groupby('CODE_GENDER')['AMT_CREDIT'].mean() 
    cred_by_family_status=df.groupby('NAME_FAMILY_STATUS')['AMT_CREDIT'].mean()
    cred_by_age_group=df.groupby('NEW_AGE_GROUP')['AMT_CREDIT'].mean()

    df['NEW_AMT_CREDIT_TO_AMT_INCOME']=df['AMT_CREDIT']/df['AMT_INCOME_TOTAL'] 
    df['NEW_AMT_CREDIT_TO_AMT_ANNUITY']=df['AMT_CREDIT']/df['AMT_ANNUITY']
    df['NEW_AMT_CREDIT_TO_AMT_GOODS_PRICE']=df['AMT_CREDIT']/df['AMT_GOODS_PRICE']
    df['NEW_AMT_CREDIT_TO_MEAN_AMT_CREDIT_BY_CONTRACT_TYPE']=df['AMT_CREDIT']/(df['NAME_CONTRACT_TYPE'].map(cred_by_contract))
    df['NEW_AMT_CREDIT_TO_MEAN_AMT_CREDIT_BY_HOUSING_TYPE']=df['AMT_CREDIT']/(df['NAME_HOUSING_TYPE'].map(cred_by_housing_type))
    df['NEW_AMT_CREDIT_TO_MEAN_AMT_CREDIT_BY_ORGANIZATION_TYPE']=df['AMT_CREDIT']/(df['ORGANIZATION_TYPE'].map(cred_by_org_type))
    df['NEW_AMT_CREDIT_TO_MEAN_AMT_CREDIT_BY_EDUCATION_TYPE']=df['AMT_CREDIT']/(df['NAME_EDUCATION_TYPE'].map(cred_by_education_type))
    df['NEW_AMT_CREDIT_TO_MEAN_AMT_CREDIT_BY_GENDER']=df['AMT_CREDIT']/(df['CODE_GENDER'].map(cred_by_gender))
    df['NEW_AMT_CREDIT_TO_MEAN_AMT_CREDIT_BY_FAMILY_STATUS']=df['AMT_CREDIT']/(df['NAME_FAMILY_STATUS'].map(cred_by_family_status))
    df['NEW_AMT_CREDIT_TO_MEAN_AMT_INCOME_BY_AGE_GROUP']=df['AMT_CREDIT']/df['NEW_AGE_GROUP'].map(cred_by_age_group)


    # INCOME related
    inc_by_contract=df.groupby('NAME_CONTRACT_TYPE')['AMT_INCOME_TOTAL'].mean() 
    inc_by_housing_type=df.groupby('NAME_HOUSING_TYPE')['AMT_INCOME_TOTAL'].mean() 
    inc_by_org_type=df.groupby('ORGANIZATION_TYPE')['AMT_INCOME_TOTAL'].mean() 
    inc_by_education_type=df.groupby('NAME_EDUCATION_TYPE')['AMT_INCOME_TOTAL'].mean() 
    inc_by_gender=df.groupby('CODE_GENDER')['AMT_INCOME_TOTAL'].mean()
    inc_by_family_status=df.groupby('NAME_FAMILY_STATUS')['AMT_INCOME_TOTAL'].mean()
    inc_by_age_group=df.groupby('NEW_AGE_GROUP')['AMT_INCOME_TOTAL'].mean()

    df['NEW_AMT_INCOME_BY_AGE_GROUP']=df['AMT_INCOME_TOTAL']/df['NEW_AGE_GROUP'].map(inc_by_age_group)
    df['NEW_AMT_INCOME_BY_CNT_CHILD']=df['AMT_INCOME_TOTAL']/(1+df['CNT_CHILDREN'])
    df['NEW_AMT_INCOME_BY_CNT_FAM_MEMBERS']=df['AMT_INCOME_TOTAL']/df['CNT_FAM_MEMBERS']
    df['NEW_AMT_INCOME_BY_AGE']=df['AMT_INCOME_TOTAL']/(df['DAYS_BIRTH']/-365)
    df['NEW_AMT_INCOME_TO_MEAN_AMT_CREDIT_BY_CONTRACT_TYPE']=df['AMT_INCOME_TOTAL']/(df['NAME_CONTRACT_TYPE'].map(inc_by_contract))
    df['NEW_AMT_INCOME_TO_MEAN_AMT_CREDIT_BY_HOUSING_TYPE']=df['AMT_INCOME_TOTAL']/(df['NAME_HOUSING_TYPE'].map(inc_by_housing_type))
    df['NEW_AMT_INCOME_TO_MEAN_AMT_CREDIT_BY_ORGANIZATION_TYPE']=df['AMT_INCOME_TOTAL']/(df['ORGANIZATION_TYPE'].map(inc_by_org_type))
    df['NEW_AMT_INCOME_TO_MEAN_AMT_CREDIT_BY_EDUCATION_TYPE']=df['AMT_INCOME_TOTAL']/(df['NAME_EDUCATION_TYPE'].map(inc_by_education_type))
    df['NEW_AMT_INCOME_TO_MEAN_AMT_CREDIT_BY_GENDER']=df['AMT_INCOME_TOTAL']/(df['CODE_GENDER'].map(inc_by_gender))
    df['NEW_AMT_CREDIT_TO_MEAN_AMT_CREDIT_BY_FAMILY_STATUS']=df['AMT_CREDIT']/(df['NAME_FAMILY_STATUS'].map(inc_by_family_status))
    df['NEW_AMT_INCOME_TO_MEAN_AMT_INCOME_BY_AGE_GROUP']=df['AMT_INCOME_TOTAL']/df['NEW_AGE_GROUP'].map(inc_by_age_group)


    # FLAG related
    # doc_flags--20 columns about documents
    # contact_flags--6 flags about contact info of client (FLAG_MOBIL, FLAG_EMAIL, etc)
    # address_flags--6 flags about address info of client (REG_REGION_NOT_LIVE_REGION, REG_REGION_NOT_WORK_REGION, etc)
    doc_flags=[i for i in df.columns if 'FLAG_DOCUMENT' in i]
    contact_flags=[i for i in df.columns if ('FLAG' in i) and (i not in doc_flags) and (i not in ('FLAG_OWN_CAR', 'FLAG_OWN_REALTY'))]
    address_flags=[i for i in df.columns if 'NOT' in i]
    flag_map={'Y':1, 'N':0}

    df['NEW_DOC_FLAG_MEAN']=df[doc_flags].mean(axis=1)
    df['NEW_DOC_FLAG_SUM']=df[doc_flags].sum(axis=1)
    df['NEW_CONTACT_FLAG_MEAN']=df[contact_flags].mean(axis=1)
    df['NEW_CONTACT_FLAG_SUM']=df[contact_flags].sum(axis=1)
    df['NEW_ADDRESS_FLAG_MEAN']=df[address_flags].mean(axis=1)
    df['NEW_ADDRESS_FLAG_SUM']=df[address_flags].sum(axis=1)
    df['NEW_OWN_CAR_REALTY_COMBINATION']=0.75*df['FLAG_OWN_REALTY'].map(flag_map)+0.25*df['FLAG_OWN_CAR'].map(flag_map)


    # AGE related
    age_by_housing_type=df.groupby('NAME_HOUSING_TYPE')['DAYS_BIRTH'].mean()
    age_by_own_realty=df.groupby('FLAG_OWN_REALTY')['DAYS_BIRTH'].mean()
    age_by_own_car=df.groupby('FLAG_OWN_CAR')['DAYS_BIRTH'].mean()

    df['NEW_AGE_TO_MEAN_AGE_BY_FLAG_OWN_REALTY']=df['DAYS_BIRTH']/(df['FLAG_OWN_REALTY'].map(age_by_own_realty))
    df['NEW_AGE_TO_MEAN_AGE_BY_FLAG_OWN_CAR']=df['DAYS_BIRTH']/(df['FLAG_OWN_CAR'].map(age_by_own_car))
    df['NEW_AGE_TO_MEAN_AGE_BY_HOUSING_TYPE']=df['DAYS_BIRTH']/(df['NAME_HOUSING_TYPE'].map(age_by_housing_type))
    df["NEW_DAYS_EMPLOYED_TO_DAYS_BIRTH"]=df['DAYS_EMPLOYED']/df['DAYS_BIRTH']
    df["NEW_DAYS_REGISTRATION_TO_DAYS_BIRTH"]=df['DAYS_REGISTRATION']/df['DAYS_BIRTH']


    # Other
    df['NEW_OWN_CAR_AGE_TO_DAYS_BIRTH']=df['OWN_CAR_AGE']/df['DAYS_BIRTH']
    df['NEW_OWN_CAR_AGE_TO_DAYS_EMPLOYED']=df['OWN_CAR_AGE']/df['DAYS_EMPLOYED']
    df['NEW_DAYS_LAST_PHONE_CHANGE_TO_DAYS_BIRTH']=df['DAYS_LAST_PHONE_CHANGE']/df['DAYS_BIRTH']
    df['NEW_DAYS_LAST_PHONE_CHANGE_TO_DAYS_EMPLOYED']=df['DAYS_LAST_PHONE_CHANGE']/df['DAYS_EMPLOYED']
    df['NEW_CNT_CHILD_TO_CNT_FAM_MEMBERS']=df['CNT_CHILDREN']/df['CNT_FAM_MEMBERS']
    df['NEW_EXT_SOURCES_MEAN']=df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    df['NEW_EXT_SOURCES_STD']=df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    df['NEW_DAYS_CHANGE_MEAN']=df[['DAYS_ID_PUBLISH', 'DAYS_LAST_PHONE_CHANGE', 'DAYS_REGISTRATION']].mean(axis=1)
    df['NEW_REGION_RATING_CLIENT_MEAN']=df[['REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY']].mean(axis=1)
    df['NEW_30_CNT_SOCIAL_CIRCLE_MEAN']=df[['OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE']].mean(axis=1)
    df['NEW_60_CNT_SOCIAL_CIRCLE_MEAN']=df[['OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE']].mean(axis=1)
    
    print(f'After adding features: {df.shape}')
    return df


def numeric_agg(df, group_col, df_name):
    num_df=df.select_dtypes('number')
    if num_df.shape[1]!=0:
        for c in num_df.columns:
            if 'ID' in c and c!=group_col:
                num_df=num_df.drop(c, axis=1)
        df_agg=num_df.groupby(group_col).agg(['count', 'mean', 'max', 'min', 'sum'])

        new_cols=[]
        for l1 in df_agg.columns.levels[0]:
            if l1!=group_col:
                for l2 in df_agg.columns.levels[1]: # for stat in agg.columns.levels[1][:-1]
                    new_cols.append(f'{df_name}_{l1}_{l2}')
        df_agg.columns=new_cols
        # Remove duplicate columns by values
        _, idx = np.unique(df_agg, axis = 1, return_index = True)
        df_agg = df_agg.iloc[:, idx]
        print(f'Dataset:{df_name}\n\tBefore: {num_df.shape[1]} numeric cols\n\tAfter: {df_agg.shape[1]}')
        return df_agg
    else:
        print('No numeric columns in dataframe')
        return False
    
    
def categ_agg(df, group_col, df_name, enc, enc_mode='train'):
    cat_df=df.select_dtypes(include=['category', 'object'])
    if cat_df.shape[1]!=0:
        if enc_mode=='train':
            cat_df_ohe=enc.fit_transform(cat_df)
        elif enc_mode=='test': 
            cat_df_ohe=enc.transform(cat_df)
        cat_df_ohe=pd.DataFrame(cat_df_ohe, columns=enc.get_feature_names_out())
        cat_df_ohe[group_col]=df[group_col]
        df_agg=cat_df_ohe.groupby(group_col).agg(['sum', 'mean'])

        new_cols=[]
        for l1 in df_agg.columns.levels[0]:
            for l2 in ['count', 'count_norm']: # more suitable aliases for sum and mean
                new_cols.append(f'{df_name}_{l1}_{l2}')
        df_agg.columns=new_cols
        # Remove duplicate columns by values
        _, idx = np.unique(df_agg, axis = 1, return_index = True)
        df_agg = df_agg.iloc[:, idx]
        print(f'Dataset:{df_name}\n\tBefore: {cat_df.shape[1]} categorical cols\n\tAfter: {df_agg.shape[1]}')
        return df_agg
    else:
        print('No categorical columns in dataframe')
        return False
    

def agg_combine(df, group_vars, df_names, enc, enc_mode='train', agg_level=1):
    if agg_level==2:
        df_cat_agg=categ_agg(df, group_vars[1], df_names[1], enc, enc_mode)
        df_num_agg=numeric_agg(df, group_vars[1], df_names[1])
        df_full_l2=df_cat_agg.merge(df_num_agg, on=group_vars[1], how='outer')
        df_full_l2=df[group_vars].merge(df_full_l2, on=group_vars[1], how='right')
        df_full=numeric_agg(df_full_l2, group_vars[0], df_names[0])
        gc.enable()
        del df_full_l2
        gc.collect()
    elif agg_level==1:
        df_cat_agg=categ_agg(df, group_vars[0], df_names[0], enc, enc_mode)
        df_num_agg=numeric_agg(df, group_vars[0], df_names[0])
        df_full=df_cat_agg.merge(df_num_agg, on=group_vars[0], how='outer')
    else:
        return 'Select aggregation level 1 or 2'
    gc.enable()
    del df_cat_agg, df_num_agg
    gc.collect()
    return df_full