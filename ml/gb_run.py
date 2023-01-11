"""
gb running code
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import os
from performance import printPerformance

# Utility functions
def drop_label(X):
    # Create label and drop ['is_canceled']
    y = X['is_canceled']
    # X.columns
    X = X.drop(columns=['is_canceled']) #Edit drop
    return X, y

def get_pipeline(categories=None):
    #numeric pipeline
    num_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), 
                                ("scaler", StandardScaler())])
    #categorical pipeline
    cat_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                ("onehot", OneHotEncoder())])
    return num_pipeline, cat_pipeline

# Load the dataset
h1_train = pd.read_csv('Input_data/h1_trainval.csv')
h1_test = pd.read_csv('Input_data/h1_test.csv')
h2_train = pd.read_csv('Input_data/h2_trainval.csv')
h2_test = pd.read_csv('Input_data/h2_test.csv')

# Drop column 'is_canceled'
X_h1_train, y_h1_train = drop_label(h1_train)
X_h2_train, y_h2_train = drop_label(h2_train)
X_h1_test, y_h1_test = drop_label(h1_test)
X_h2_test, y_h2_test = drop_label(h2_test)

# Classify columns type
object_columns = X_h1_train.select_dtypes(include=[object]).columns
num_columns = X_h1_train.select_dtypes(include=[int,float]).columns
num_ls = [x for x in num_columns]
cat_ls = [x for x in object_columns]

# Model selection
classifier = GradientBoostingClassifier()
model = 'gb'
n_estimators = [50, 100, 150, 200]
max_features = np.arange(0.2, 0.95, 0.05)
min_samples_split = np.arange(2, 10)
learning_rate = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.75, 1]
parameters_grids = {
    'n_estimators': n_estimators, 
    'max_features': max_features,
    'min_samples_split': min_samples_split,
    'learning_rate': learning_rate
}
### H1 training ###
trials = []
aucroc_ls, aucpr_ls, acc_ls, ba_ls, sen_ls, spe_ls, pre_ls, mcc_ls, f1_ls, ck_ls = [], [], [], [], [], [], [], [], [], []
for t in range(5): # Replication 
    #set up transformer to standardize all vars
    col_transform = ColumnTransformer([("num", get_pipeline()[0], num_ls),
                                        ("cat", get_pipeline()[1], cat_ls)
                                    ])
    #create model pipeline
    pipe = Pipeline([('transformer', col_transform), ('clf', classifier)])
    #gridsearch cv
    grid_cv = GridSearchCV(pipe, 
                            parameters_grids, 
                            scoring="roc_auc",
                            n_jobs=-1,
                            cv=TimeSeriesSplit(n_splits=5, gap=t*10),
                            return_train_score=True)
    grid_cv.fit(X_h1_train, y_h1_train) 
    pred_proba = pipe.predict_proba(X_h1_test)
    aucroc, aucpr, acc, ba, sen, spe, pre, mcc, f1, ck = printPerformance(y_h1_test, pred_proba[:,1], auc_only=False)
    aucroc_ls.append(aucroc)
    aucpr_ls.append(aucpr)
    acc_ls.append(acc)
    ba_ls.append(ba)
    sen_ls.append(sen)
    spe_ls.append(spe)
    pre_ls.append(pre)
    mcc_ls.append(mcc)
    f1_ls.append(f1)
    ck_ls.append(ck)
    trials.append(t)
    d = {'auc_roc':aucroc_ls,'aucpr':aucpr_ls,'acc':acc_ls,'ba_acc':ba_ls,
    'sen':sen_ls,'spe':spe_ls,'pre':pre_ls,'mcc':mcc_ls,'f1':f1_ls,'ck':ck_ls,}
    scores_df = pd.DataFrame(d,
                            columns=['auc_roc', 'aucpr', 'acc', 'ba_acc', 'sen', 'spe', 'pre', 'mcc', 'f1', 'ck'],
                            index = trials)
scores_df.to_csv(os.path.join('output', f"{model}_H1.csv"))

### H2 training ###
trials = []
aucroc_ls, aucpr_ls, acc_ls, ba_ls, sen_ls, spe_ls, pre_ls, mcc_ls, f1_ls, ck_ls = [], [], [], [], [], [], [], [], [], []
for t in range(5): # Replication 
    #set up transformer to standardize all vars
    col_transform = ColumnTransformer([("num", get_pipeline()[0], num_ls),
                                        ("cat", get_pipeline()[1], cat_ls)
                                    ])
    #create model pipeline
    pipe = Pipeline([('transformer', col_transform), ('clf', classifier)])
    #gridsearch cv
    grid_cv = GridSearchCV(pipe, 
                            parameters_grids, 
                            scoring="roc_auc",
                            n_jobs=-1,
                            cv=TimeSeriesSplit(n_splits=5, gap=t*10),
                            return_train_score=True)
    grid_cv.fit(X_h2_train, y_h2_train) 
    pred_proba = pipe.predict_proba(X_h2_test)
    aucroc, aucpr, acc, ba, sen, spe, pre, mcc, f1, ck = printPerformance(y_h2_test, pred_proba[:,1], auc_only=False)
    aucroc_ls.append(aucroc)
    aucpr_ls.append(aucpr)
    acc_ls.append(acc)
    ba_ls.append(ba)
    sen_ls.append(sen)
    spe_ls.append(spe)
    pre_ls.append(pre)
    mcc_ls.append(mcc)
    f1_ls.append(f1)
    ck_ls.append(ck)
    trials.append(t)
    d = {'auc_roc':aucroc_ls,'aucpr':aucpr_ls,'acc':acc_ls,'ba_acc':ba_ls,
    'sen':sen_ls,'spe':spe_ls,'pre':pre_ls,'mcc':mcc_ls,'f1':f1_ls,'ck':ck_ls,}
    scores_df = pd.DataFrame(d,
                            columns=['auc_roc', 'aucpr', 'acc', 'ba_acc', 'sen', 'spe', 'pre', 'mcc', 'f1', 'ck'],
                            index = trials)
scores_df.to_csv(os.path.join('output', f"{model}_H2.csv"))




