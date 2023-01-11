import torch
from torch.utils.data import Dataset,DataLoader
from training_process import train

class Load_data(Dataset):
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

def run_TNN(X_train,X_test,y_train,y_test,model,
              criterion,optimizer,batch_size, epochs, model_name):
    '''
    run_TNN actually executes the entire training and validation methods for the models. Prints the metrics for the task
    and plots the graphs of accuracy vs epochs and loss vs epochs.
         :param X_train(numpy array): train data
         :param y_train(numpy array): Target variable of the training data
         :param X_test(numpy array): test data
         :param y_test(numpy array): Target variable of the test data
         :param model(TNN Classifier): model to be trained
         :param criterion(object of loss function): Loss function to be used for training
         :param optimizer(object of Optimizer): Optimizer used for training
         :param batch_size(int,optional): Batch size used for training and validation.
         :param epochs(int,optional): Number of epochs for training the model.
      :return:
         model object, list of training accuracy, training loss, testing accuracy, testing loss for all the epochs
    '''
    trainDataload = DataLoader(Load_data(X_train, y_train), batch_size=batch_size)
    testDataload = DataLoader(Load_data(X_test, y_test), batch_size=batch_size)
    best_model, train_loss_ls, val_loss_ls = train(model, trainDataload, testDataload, criterion, optimizer , epochs, model_name)
    return best_model, train_loss_ls, val_loss_ls