import torch
import numpy as np
from xgboost import XGBClassifier,XGBRegressor
from collections import OrderedDict
from tree_process import tree_process
import random

class TNN_Classifier(torch.nn.Module):
    '''
    TNNClassifier is a model for classification tasks that tries to combine tree-based models with
    neural networks to create a robust architecture.
         :param X_values(numpy array): Train data
         :param y_values(numpy array): Label data
         :param num_layers(int): Number of layers in the neural network
         :param num_layers_boosted(int,optional): Number of layers to be boosted in the neural network. Default value: 1
         :param kind_layers(int): Kind of model to train
    '''
    def __init__(self, X_values, y_values, num_layers, num_layers_boosted=1,
                 kind_layers = 32):
        super(TNN_Classifier, self).__init__()
        np.random.seed(1)
        random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        self.name = "Classification"
        self.layers = OrderedDict()
        self.boosted_layers = {}
        self.num_layers = num_layers
        self.num_layers_boosted = num_layers_boosted
        self.kind_layers = kind_layers
        self.X = X_values
        self.y = y_values

        self.take_layers_dim()
        self.base_tree()

        self.layers[str(0)].weight = torch.nn.Parameter(torch.from_numpy(self.temp.T))

        self.xg = XGBClassifier(n_estimators=100)

        self.tree_process = tree_process(self.layers)
        self.tree_process.give(self.xg, self.num_layers_boosted)
        self.feature_importances_ = None
        
    def get(self, l):
        '''
        Gets the set of current actual outputs of the inputs
        :param l(tensor): Labels of the current set of inputs that are getting processed.
        '''
        self.l = l

    def take_layers_dim(self):
        # Model of 32 dimensions
        if self.kind_layers == 32: 
            self.layers[0] = torch.nn.Linear(28, 32, bias=False)
            self.layers[0] = torch.nn.LeakyReLU()
            self.layers[1] = torch.nn.Linear(32, 32, bias=False)
            self.layers[1] = torch.nn.LeakyReLU()
            self.layers[2] = torch.nn.Linear(32, 1, bias=False)
            self.layers[2] = torch.nn.Sigmoid()
        
        # Model of 64 dimensions
        elif self.kind_layers == 64:
            self.layers[0] = torch.nn.Linear(28, 64, bias=False)
            self.layers[0] = torch.nn.LeakyReLU()
            self.layers[1] = torch.nn.Linear(64, 32, bias=False)
            self.layers[1] = torch.nn.LeakyReLU()
            self.layers[2] = torch.nn.Linear(32, 1, bias=False)
            self.layers[2] = torch.nn.Sigmoid()
        
        # Model of 128 dimensions
        elif self.kind_layers == 128:
            self.layers[0] = torch.nn.Linear(28, 128, bias=False)
            self.layers[0] = torch.nn.LeakyReLU()
            self.layers[1] = torch.nn.Linear(128, 64, bias=False)
            self.layers[1] = torch.nn.LeakyReLU()
            self.layers[2] = torch.nn.Linear(64, 1, bias=False)
            self.layers[2] = torch.nn.Sigmoid()

    def base_tree(self):
        '''
        Instantiates and trains a XGBClassifier on the first layer of the neural network to set its feature importances
         as the weights of the layer
        '''
        self.temp1 = XGBClassifier(n_estimators=100).fit(self.X, self.y,eval_metric="mlogloss").feature_importances_
        self.temp = self.temp1
        for i in range(1, self.input_out_dim):
            self.temp = np.column_stack((self.temp, self.temp1))

    def forward(self, x, train=True):
        x = self.tree_process(x, self.l,train)
        return x

    def save(self,path):
        '''
        Saves the entire model in the provided path
        :param path(string): Path where model should be saved
        '''
        torch.save(self,path)
