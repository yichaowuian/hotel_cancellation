
import torch
from TNN.training_process import predict_proba
from TNN.models import TNN_Classifier
from TNN.main import run_TNN
from TNN.utils import load_dataset
import os 
from dl_base.performance import printPerformance

# Load dataset
X_h1_train, y_h1_train, X_h1_test, y_h1_test = load_dataset('Input_data/h1_dl_train.csv','Input_data/h1_dl_test.csv')
X_h2_train, y_h2_train, X_h2_test, y_h2_test = load_dataset('Input_data/h2_dl_train.csv','Input_data/h2_dl_test.csv')

# Set parameter
lr = 0.00005
batch_size = 128
epochs = 100
num_boosted = 3

### H1 ###
# Create folder for saving output and checkpoints
name = f'H1_TNN_32'
if not os.path.exists(f'./checkpoints/{name}'):
    os.makedirs(f'./checkpoints/{name}')
if not os.path.exists(f'./output/{name}'):
    os.makedirs(f'./output/{name}')

# Loading model
model = TNN_Classifier(X_h1_train,y_h1_train,3, num_layers_boosted = num_boosted, kind_layers=32)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Start training
best_model_h1, train_loss_ls, test_loss_ls = run_TNN(X_h1_train, X_h1_test, y_h1_train, y_h1_test, model, criterion, optimizer,batch_size,epochs,name)

# Prediction
pred_prob_h1 = predict_proba(best_model_h1, X_h1_test)
aucroc, aucpr, acc, ba, sen, spe, pre, mcc, f1, ck = printPerformance(y_h1_test, pred_prob_h1.detach().numpy()[:,1], printout=False, auc_only=False)
with open(f'./output/{name}/{name}_e{2}_test.txt', 'w') as f:
        f.write(f'aucroc:{aucroc}, aucpr:{aucpr}, acc:{acc}, ba:{ba}, sen:{sen}, spe:{spe}, pre:{pre}, mcc:{mcc}, f1:{f1}, ck:{ck}')

### H2 ###
# Create folder for saving output and checkpoints
name = f'H2_TNN_32'
if not os.path.exists(f'./checkpoints/{name}'):
    os.makedirs(f'./checkpoints/{name}')
if not os.path.exists(f'./output/{name}'):
    os.makedirs(f'./output/{name}')

# Loading model
model = TNN_Classifier(X_h2_train,y_h2_train,3, num_layers_boosted = num_boosted, kind_layers=32)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# Start training
best_model_h2, train_loss_ls, test_loss_ls = run_TNN(X_h2_train, X_h2_test, y_h2_train, y_h2_test, model, criterion, optimizer,batch_size,epochs,name)

# Prediction
pred_prob_h2 = predict_proba(best_model_h2, X_h2_test)
aucroc, aucpr, acc, ba, sen, spe, pre, mcc, f1, ck = printPerformance(y_h2_test, pred_prob_h2.detach().numpy()[:,1], printout=False, auc_only=False)
with open(f'./output/{name}/{name}_e{2}_test.txt', 'w') as f:
        f.write(f'aucroc:{aucroc}, aucpr:{aucpr}, acc:{acc}, ba:{ba}, sen:{sen}, spe:{spe}, pre:{pre}, mcc:{mcc}, f1:{f1}, ck:{ck}')
