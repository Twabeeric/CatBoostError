# -*- coding: utf-8 -*-
"""


"""

import lightgbm as lgb 
import pandas as pd
import os
import numpy as np
import random
import scipy
import gc
import time


random.seed(20)

 #Loading Data	
df_temp1 = pd.read_csv('/content/drive/My Drive/Colab Notebooks/train_transaction.csv')
df_temp2 = pd.read_csv('/content/drive/My Drive/Colab Notebooks/train_identity.csv')
df_train1=pd.merge(df_temp1, df_temp2, left_on='TransactionID', right_on='TransactionID', how='outer')

df_temp3 = pd.read_csv('/content/drive/My Drive/Colab Notebooks/test_transaction.csv')
df_temp4 = pd.read_csv('/content/drive/My Drive/Colab Notebooks/test_identity.csv')
df_test1=pd.merge(df_temp3, df_temp4, left_on='TransactionID', right_on='TransactionID', how='outer')

del df_temp1,df_temp2, df_temp3, df_temp4

#
df_train1['Hour'] = np.floor((df_train1['TransactionDT']/3600)%24)
df_train1['Week'] = np.floor((df_train1['TransactionDT']/3600/24))
df_test1['Hour'] = np.floor((df_test1['TransactionDT']/3600)%24)
df_test1['Week'] = np.floor((df_test1['TransactionDT']/3600/24))
df_train1['DecimalP'] = abs(np.floor(np.log10(df_train1['TransactionAmt'])))
df_train1.loc[np.isinf(df_train1['DecimalP']), 'DecimalP'] = 0
df_new = df_train1["P_emaildomain"].str.split(".", n=1, expand=True)
df_train1['PDomain'] = df_new[0]
df_train1['PCountry'] = df_new[1]
df_new = df_train1["R_emaildomain"].str.split(".", n=1, expand=True)
df_train1['RDomain'] = df_new[0]
df_train1['RCountry'] = df_new[1]

df_test1['DecimalP'] =abs(np.floor(np.log10(df_test1['TransactionAmt'])))
df_test1.loc[np.isinf(df_test1['DecimalP']), 'DecimalP'] = 0
df_new = df_test1["P_emaildomain"].str.split(".", n=1, expand=True)
df_test1['PDomain'] = df_new[0]
df_test1['PCountry'] = df_new[1]
df_new = df_test1["R_emaildomain"].str.split(".", n=1, expand=True)
df_test1['RDomain'] = df_new[0]
df_test1['RCountry'] = df_new[1]


df_new = df_train1["id_30"].str.split(" ", expand=True)
df_train1['OS'] = df_new[0]
df_new = df_test1["id_30"].str.split(" ", expand=True)
df_test1['OS'] = df_new[0]
df_new = df_train1["id_31"].str.split(" ", expand=True)
df_train1['Browser'] = df_new[0]
df_new = df_test1["id_31"].str.split(" ", expand=True)
df_test1['Browser'] = df_new[0]


df_train1['uniqueaddr']=(df_train1.addr1*100)+df_train1.addr2
df_test1['uniqueaddr']=(df_test1.addr1*100)+df_test1.addr2

df_new = df_train1["DeviceInfo"].str.split(" ", expand=True)
df_train1['Device'] = df_new[0]
df_new = df_test1["DeviceInfo"].str.split(" ", expand=True)
df_test1['Device'] = df_new[0]


df_train1['AmtDst']=(df_train1.TransactionAmt*df_train1.dist1)/1000
df_test1['AmtDst']=(df_test1.TransactionAmt*df_test1.dist1)/1000


df_train1['PDomain'].replace(['ymail'], 'yahoo')
df_train1['PDomain'].replace(['bellsouth'], 'att')
df_train1['PDomain'].replace(['msn'], 'outlook')
df_train1['PDomain'].replace(['frontiernet'], 'frontier')

df_test1['PDomain'].replace(['ymail'], 'yahoo')
df_test1['PDomain'].replace(['bellsouth'], 'att')
df_test1['PDomain'].replace(['msn'], 'outlook')
df_test1['PDomain'].replace(['frontiernet'], 'frontier')

df_train1['RDomain'].replace(['ymail'], 'yahoo')
df_train1['RDomain'].replace(['bellsouth'], 'att')
df_train1['RDomain'].replace(['msn'], 'outlook')
df_train1['RDomain'].replace(['frontiernet'], 'frontier')

df_test1['RDomain'].replace(['ymail'], 'yahoo')
df_test1['RDomain'].replace(['bellsouth'], 'att')
df_test1['RDomain'].replace(['msn'], 'outlook')
df_test1['RDomain'].replace(['frontiernet'], 'frontier')

df_train1['PCountry'].replace(['net.mx'], 'com.mx')
df_train1['RCountry'].replace(['net.mx'], 'com.mx')

df_test1['PCountry'].replace(['net.mx'], 'com.mx')
df_test1['RCountry'].replace(['net.mx'], 'com.mx')

df_train1['AmtFlag']=np.where((df_train1['TransactionAmt'] >= df_train1['TransactionAmt'].mean()),1,0)
df_test1['AmtFlag']=np.where((df_test1['TransactionAmt'] >= df_test1['TransactionAmt'].mean()),1,0)

df_train1['uid1']=(df_train1.card1/1000)*(df_train1.card2/100)
df_train1['uid2']=(df_train1.card1/1000)*(df_train1.card3/100)
df_train1['uid3']=(df_train1.card1/1000)*(df_train1.card2/100)*(df_train1.card3/100)*(df_train1.card5/100)


df_test1['uid1']=(df_test1.card1/1000)*(df_test1.card2/100)
df_test1['uid2']=(df_test1.card1/1000)*(df_test1.card3/100)
df_test1['uid3']=(df_test1.card1/1000)*(df_test1.card2/100)*(df_test1.card3/100)*(df_test1.card5/100)
##Transforming TransactionAmt
df_train1['TransactionAmt'] = np.log1p(df_train1['TransactionAmt'])
df_test1['TransactionAmt'] = np.log1p(df_test1['TransactionAmt'])

df_train1['TAmtFq']=0
df_test1['TAmtFq']=0

for c in range(len(df_train1)):
  if df_train1.ix[c,'TAmtFq']==0:
    df_train1.ix[c,'TAmtFq']=(df_train1['TransactionAmt']== df_train1.TransactionAmt[c]).sum()
    df_train1.loc[df_train1.TransactionAmt==df_train1.TransactionAmt[c],'TAmtFq']=(df_train1['TransactionAmt']== df_train1.TransactionAmt[c]).sum()
for c in range(len(df_test1)):
  if df_test1.ix[c,'TAmtFq']==0:
    df_test1.ix[c,'TAmtFq']=(df_test1['TransactionAmt']== df_test1.TransactionAmt[c]).sum()
    df_test1.loc[df_test1.TransactionAmt==df_test1.TransactionAmt[c],'TAmtFq']=(df_test1['TransactionAmt']== df_test1.TransactionAmt[c]).sum()


df_train1['Duplicate']=0
df_test1['Duplicate']=0
for d in range(len(df_train1)-5):
  if d>=5:
    if df_train1.TransactionAmt[d]==df_train1.TransactionAmt[d+1] or df_train1.TransactionAmt[d]==df_train1.TransactionAmt[d+2] or df_train1.TransactionAmt[d]==df_train1.TransactionAmt[d+3] or df_train1.TransactionAmt[d]==df_train1.TransactionAmt[d+4] or df_train1.TransactionAmt[d]==df_train1.TransactionAmt[d+5] or df_train1.TransactionAmt[d]==df_train1.TransactionAmt[d-1] or df_train1.TransactionAmt[d]==df_train1.TransactionAmt[d-2] or df_train1.TransactionAmt[d]==df_train1.TransactionAmt[d-3] or df_train1.TransactionAmt[d]==df_train1.TransactionAmt[d-4] or df_train1.TransactionAmt[d]==df_train1.TransactionAmt[d-5]:
      df_train1.ix[d,'Duplicate']=1
    
for d in range(len(df_test1)-5):
  if d>=5:
    if df_test1.TransactionAmt[d]==df_test1.TransactionAmt[d+1] or df_test1.TransactionAmt[d]==df_test1.TransactionAmt[d+2] or df_test1.TransactionAmt[d]==df_test1.TransactionAmt[d+3] or df_test1.TransactionAmt[d]==df_test1.TransactionAmt[d+4] or df_test1.TransactionAmt[d]==df_test1.TransactionAmt[d+5] or df_test1.TransactionAmt[d]==df_test1.TransactionAmt[d-1] or df_test1.TransactionAmt[d]==df_test1.TransactionAmt[d-2] or df_test1.TransactionAmt[d]==df_test1.TransactionAmt[d-3] or df_test1.TransactionAmt[d]==df_test1.TransactionAmt[d-4] or df_test1.TransactionAmt[d]==df_test1.TransactionAmt[d-5]:
      df_test1.ix[d,'Duplicate']=1
      
df_train1['nulls1'] = df_train1.isna().sum(axis=1)
df_test1['nulls1'] = df_test1.isna().sum(axis=1)

catlist =['ProductCD','card1','card2','card3','card4', 'card5', 'card6','addr1','addr2', 'M1',
       'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18',
              'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_29',
              'id_30', 'id_31','id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType','P_emaildomain', 'R_emaildomain', 'DeviceInfo',
       'PDomain', 'PCountry', 'RDomain', 'RCountry', 'OS', 'Browser', 'uniqueaddr', 'Device']


del df_new



gc.collect()

#Drop columns with high no of nulls
def missing(dff):
    print (round((dff.isnull().sum() * 100/ len(dff)),2).sort_values(ascending=False))
    xmiss=round((dff.isnull().sum() * 100/ len(dff)),2).sort_values(ascending=False)
    return xmiss
ls1=pd.DataFrame(missing(df_train1))

def rmissingvaluecol(dff,threshold):
    l = []
    l = list(dff.drop(dff.loc[:,list((100*(dff.isnull().sum()/len(dff.index))>=threshold))].columns, 1).columns.values)
    z=list(set(list((dff.columns.values))) - set(l))
    return z
ls2=rmissingvaluecol(df_train1[df_train1.isFraud==1],85)

### Duplicates
duplicates = []
cols = df_train1.columns
i = 0
for c1 in cols:
    i += 1
    for c2 in cols[i:]:
        if c1 != c2:
            if (np.sum((df_train1[c1].values == df_train1[c2].values).astype(int)) / len(df_train1))>0.95:
                duplicates.append(c2)
                print(c1, c2, np.sum((df_train1[c1].values == df_train1[c2].values).astype(int)) / len(df_train1))
print(list(set(duplicates)))
ls3=list(set(duplicates))     
   
gc.collect()

for df in [df_train1, df_test1]:
    df['DeviceInfo'] = df['DeviceInfo'].fillna('unknown_device').str.lower()
    df['DeviceInfo_device'] = df['DeviceInfo'].apply(lambda x: ''.join([i for i in x if i.isalpha()]))
    df['DeviceInfo_version'] = df['DeviceInfo'].apply(lambda x: ''.join([i for i in x if i.isnumeric()]))

catlist.append('DeviceInfo_device')
catlist.append('DeviceInfo_version')

#Handling for features common between train/test
for col in catlist: 
    valid_card = pd.concat([df_train1[[col]], df_test1[[col]]])
    valid_card = valid_card[col].value_counts()
    valid_card = valid_card[valid_card>2]
    valid_card = list(valid_card.index)

    df_train1[col] = np.where(df_train1[col].isin(df_test1[col]), df_train1[col], np.nan)
    df_test1[col]  = np.where(df_test1[col].isin(df_train1[col]), df_test1[col], np.nan)

    df_train1[col] = np.where(df_train1[col].isin(valid_card), df_train1[col], np.nan)
    df_test1[col]  = np.where(df_test1[col].isin(valid_card), df_test1[col], np.nan)
gc.collect()

#@title

#Dealing with categorical features length >440
cutdown_ls=[]
for col, values in df_test1[catlist].iteritems():
  num_uniques = values.nunique()
  #print ('{name}: {num_unique}'.format(name=col, num_unique=num_uniques))
  if (values.nunique()>=440)==True:
      cutdown_ls.append(col)
#  print (values.unique())
  #print ('\n')
cutdown_ls

#@title
print(len(df_train1.card1.unique()))
print(len(df_train1.card2.unique()))
print(len(valid_card))

#Removing columns with noise 

rm_ls=['TransactionID','isFraud','TransactionDT','uid1','uid2','uid3']

ls_keep=[x for x in df_train1.columns if x not in rm_ls]

######################################################################################################
#Normalizing columns
#num_ls=[x for x in df_train1.columns if x not in rm_ls if x not in catlist]
#for col in num_ls:
    #df_train1[col] = ( df_train1[col]-df_train1[col].mean() ) / df_train1[col].std() 
    #df_test1[col] = ( df_test1[col]-df_test1[col].mean() ) / df_test1[col].std() 
#gc.collect()

df_train1.replace(-999, np.nan)
df_test1.replace(-999, np.nan)
[df_train1[c].fillna(-999, inplace=True) for c in df_train1.columns[df_train1.isnull().any()]]
[df_test1[c].fillna(-999, inplace=True) for c in df_test1.columns[df_test1.isnull().any()]]  

gc.collect()



#Creating equivalent subsample
from sklearn.utils import shuffle
X_train=df_train1[df_train1.isFraud==1]
X_train=pd.concat([X_train,df_train1[df_train1.isFraud==0].iloc[:(len(df_train1[df_train1.isFraud==1])),]],axis=0).sample(frac=1, random_state=42)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

for c in catlist: 
  le.fit(list(X_train[c].values))
  X_train[c]=le.transform(list(X_train[c].values))


X = X_train[ls_keep]
y = X_train.isFraud
del X_train
gc.collect()



#RFECV
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.metrics import roc_auc_score

# Build a classification task using 3 informative features
gbmrfe = lgb.LGBMClassifier(
    max_bin =63,
    max_depth=6,
    num_leaves = 70,
    num_iterations = 500,
    min_child_weight= 0.03,
    feature_fraction = 0.4,
    bagging_fraction= 0.4,
    min_data_in_leaf= 1,
    objective= 'binary',
    learning_rate= 0.01,
    boosting_type= "gbdt",
    bagging_seed= 11,
    metric= 'auc',
    random_state= 47,
    num_thread = -1,
)

gc.collect()

rfecv = RFECV(estimator=gbmrfe, step=1, cv=StratifiedKFold(2),min_features_to_select=1, verbose=5,scoring='roc_auc')
# %time rfecv.fit(X,y)
gc.collect()
print("Optimal number of features : %d" % rfecv.n_features_)
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
#Creating ranking and weights
df_ranking=pd.DataFrame(X.columns,columns=['fea'])
df_ranking['rank']=rfecv.ranking_
df_ranking.head


df_ranking.to_csv('/content/drive/My Drive/Colab Notebooks/ranking1.csv', index=False)

gc.collect()

new_cat=[x for x in catlist if x not in rm_ls]
cat_featuresls=[]
for i in new_cat:
    cat_featuresls.append(df_train1[ls_keep].columns.get_loc(i))

print(len(catlist))
print(len(new_cat))

del X,y

pip install catboost


rfe_cat=[x for x in catlist if x not in rm_ls if x in df_train1.columns]
cat_featuresls=[]
for i in rfe_cat:
    cat_featuresls.append(df_train1.columns.get_loc(i))
######################################################################################################################################
from catboost import CatBoostClassifier, Pool, cv
##############################################################################################


train_data = Pool(data=df_train1[ls_keep].ix[:round(len(df_train1)*.8),:],
                  label=df_train1.isFraud[:round(len(df_train1)*.8)+1],
                  cat_features=cat_featuresls,
                  thread_count=-1)
test_data = Pool(data=df_train1[ls_keep].ix[round(len(df_train1)*.8):,:],
                  label=df_train1.isFraud[round(len(df_train1)*.8):],
                  cat_features=cat_featuresls,
                  thread_count=-1)
gc.collect()
##############################################################################################

#########################################################################################

simple_model = CatBoostClassifier(
    loss_function= 'CrossEntropy',
    task_type= 'CPU',
    early_stopping_rounds=100,
    random_state= 47,
    use_best_model=False,
    eval_metric='AUC',
    thread_count=-1,
    verbose=200,
    l2_leaf_reg=9,
    learning_rate=.1
    )

# %time simple_model.fit(train_data,plot=True,eval_set=test_data)

############################################################################################

# Commented out IPython magic to ensure Python compatibility.
rfe_cat1=[x for x in catlist if x in df_test1[ls_keep].columns if x not in rm_ls ]
cat_featuresls1=[]
for i in rfe_cat1:
    cat_featuresls1.append(df_test1[ls_keep].columns.get_loc(i))


simple_model1 = CatBoostClassifier(
    loss_function= 'CrossEntropy',
    task_type= 'CPU',
    early_stopping_rounds=100,
    random_state= 47,
    use_best_model=False,
    eval_metric='AUC',
    max_depth=9,
    thread_count=-1,
    verbose=200,
    l2_leaf_reg=9,
    iterations=20000,
    learning_rate=.01
    )
train_data1 = Pool(data=df_train1[ls_keep],
                  label=df_train1.isFraud,
                  cat_features=cat_featuresls,
                  thread_count=-1)
test_data1 = Pool(data=df_test1[ls_keep],
                  cat_features=cat_featuresls1,
                  thread_count=-1)
# %time simple_model1.fit(train_data1)

df_pred = pd.read_csv('/content/drive/My Drive/Colab Notebooks/sample_submission.csv')
df_pred["isFraud"]=simple_model1.predict_proba(test_data1,thread_count=-1,verbose=1)[:,1]

df_pred.to_csv('/content/drive/My Drive/Colab Notebooks/submissioncatboost.csv', index=False)

#Feature Evaluation
######################################################################################

import catboost
from catboost import CatBoostClassifier, Pool
from catboost.eval.catboost_evaluation import *
from plotly.offline import iplot, init_notebook_mode

from catboost import datasets
from catboost.utils import create_cd

#train_df, _ = datasets.amazon()

dataset_dir = os.path.join('/content/drive/My Drive/Colab Notebooks/')
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

train_file = os.path.join(dataset_dir, 'train1ready.csv')
description_file = os.path.join(dataset_dir, 'train.cd')
train_df = pd.read_csv('/content/drive/My Drive/Colab Notebooks/train1ready.csv')


catlist =['ProductCD','card1','card2','card3','card4', 'card5', 'card6','addr1','addr2', 'M1',
       'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18',
              'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_29',
              'id_30', 'id_31','id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType','P_emaildomain', 'R_emaildomain', 'DeviceInfo',
       'PDomain', 'PCountry', 'RDomain', 'RCountry', 'OS', 'Browser', 'uniqueaddr', 'Device']



catlist.append('DeviceInfo_device')
catlist.append('DeviceInfo_version')
cat_featuresls=[]
for i in catlist:
    cat_featuresls.append(train_df.columns.get_loc(i))

#description_file = ('/content/drive/My Drive/Colab Notebooks/train.cd', index=False) 

#train_df.to_csv(train_file, header=False, index=False)

feature_names1 = dict()
for column, name in enumerate(train_df):
    if column == 0:
        continue
    feature_names1[column-1] = name
    
create_cd( label=0, 
           cat_features=cat_featuresls,
           feature_names=feature_names1,
           output_path=os.path.join(dataset_dir, 'train.cd')
           )
                          
fold_size = 200000
fold_offset = 0
folds_count = 20
random_seed = 0



learn_params = {'iterations': 2000, 
                'random_seed': 0, 
                'logging_level': 'Silent',
                'loss_function': 'Logloss',
                # You could set learning process to GPU
                # 'devices': '1',  
                # 'task_type': 'GPU',
                #'loss_function' : 'Logloss',
                'boosting_type': 'Plain', 
                # For feature evaluation learning time is important and we need just the relative quality
                'max_ctr_complexity' : 4,
               'target_border':.3}



#features_to_evaluate = 
#for i in catlist:
#  features_to_evaluate.append(train_df.columns.get_loc(i))
features_to_evaluate = [5, 6, 7]
from os.path import join

evaluator = CatboostEvaluation(train_file,
                               fold_size,
                               folds_count,
                               delimiter=',',
                               column_description=description_file,
                               partition_random_seed=random_seed,
                               #working_dir=...  — working directory, we will need to create temp files during evaluation, 
                               #so ensure you have enough free space. 
                               #By default we will create unique temp dir in system temp directory
                               #group_column=... — set it if you have column which should be used to split 
)



result = evaluator.eval_features(learn_config=learn_params,
                                 eval_metrics=["Logloss", "Accuracy"],
                                 features_to_eval=features_to_evaluate)

logloss_result = result.get_metric_results("Logloss")

logloss_result.get_baseline_comparison()
