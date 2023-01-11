"""
Utilty function
"""
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

# Define a function to convert the data to integer dtype
def to_float(data, labels):
    return data.to(torch.float), labels

def lowest_value_of_column(col):
    return col.min()

def drop_label(X):
    # Create label and drop ['is_canceled']
    y = X['is_canceled']
    # X.columns
    X = X.drop(columns=['is_canceled']) #Edit drop
    return X, y

def std_scaler(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def load_dataset(path_train, path_test):
    # Loading dataset
    data_train = pd.read_csv(path_train)
    data_test = pd.read_csv(path_test)
    # Create train and label 
    X_train, y_train = drop_label(data_train)
    X_test, y_test = drop_label(data_test)
    # df to np
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()
    # Standard Scalers
    X_train, X_test = std_scaler(X_train, X_test)
    return X_train, y_train, X_test, y_test

def eval_score(y_predicts, labels):
    auc_score = roc_auc_score(labels, y_predicts)
    prauc_score = average_precision_score(labels, y_predicts)
    return auc_score, prauc_score

def save_eval_score(auc_score, prauc_score, path_save):
    with open(path_save, 'w') as f:
        i = 0
        for auc, prauc in zip(auc_score, prauc_score):
            i += 1
            f.write(f'Epoch:{i} - AUC score: {str(auc)} - PR_AUC score: {str(prauc)} \n')

def plot_loss(train_loss, val_loss, graph_name='Training_graphs.png', save=False):
    figure, axis = plt.subplots()
    figure.suptitle('Performance of TNN')
    axis.plot(train_loss, label="Training Loss")
    axis.plot(val_loss, label="Val Loss")
    axis.set_xlabel('Epochs')
    axis.set_ylabel('Loss value')
    axis.set_title("Loss")
    axis.legend()
    if save:
        plt.savefig(graph_name)
        # plt.show()
    else:
        plt.show()

# Define a custom dataset class
class Data(Dataset):
    '''
    Dataset class for loading the data into dataloader
        :param X(numpy array): array of features
        :param y(numpy array): array of labels of the features
    '''
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

