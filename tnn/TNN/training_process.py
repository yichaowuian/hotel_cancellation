import numpy as np
from sklearn.metrics import classification_report,r2_score,mean_absolute_error,mean_squared_error,mean_squared_log_error
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score


def train(model,trainDataload,testDataload,criterion,optimizer,epochs,model_name):
    '''
    Training function for training the model with the given data
    :param model(TNN Classifier/Regressor): model to be trained
    :param trainDataload(object of DataLoader): DataLoader with training data
    :param testDataload(object of DataLoader): DataLoader with testing data
    :param criterion(object of loss function): Loss function to be used for training
    :param optimizer(object of Optimizer): Optimizer used for training
    :param epochs(int,optional): Number of epochs for training the model. Default value: 100
    :return:
    list of training accuracy, training loss, testing accuracy, testing loss for all the epochs
    '''
    ################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    ################
    accuracy = []
    train_loss_ls, val_loss_ls = [], []
    for epoch in tqdm(range(epochs),desc="Percentage training completed: "):
        running_loss = 0
        pred_prob, labels = [], []
        predictions = []
        correct = 0
        total = 0
        loss = None
        for inp, out in trainDataload:
            ################
            inp, out = inp.to(device), out.to(device)
            ################
            try:
                if out.shape[0] >= 1:
                    out = torch.squeeze(out, 1)
            except:
                pass
            model.get(out.float())
            y_pred = model(inp.float())
            if model.labels == 1:
                loss = criterion(y_pred, out.view(-1, 1).float())
            else:
                loss = criterion(y_pred, out.long())
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            for i, p in enumerate(model.parameters()):
                if i < model.num_layers_boosted:
                    l0 = torch.unsqueeze(model.sequential.boosted_layers[i], 1)
                    lMin = torch.min(p.grad)
                    lPower = torch.log(torch.abs(lMin))
                    if lMin != 0:
                        l0 = l0 * 10 ** lPower
                        p.grad += l0
                    else:
                        pass
                else:
                    pass
            outputs = model(inp.float(),train = False)
            predicted = outputs
            total += out.float().size(0)
            if model.name == "Regression":
                pass
            else:
                if model.labels == 1:
                    for i in range(len(predicted)):
                        if predicted[i] < torch.Tensor([0.5]):
                            predicted[i] = 0
                        else:
                            predicted[i] =1

                        if predicted[i].type(torch.LongTensor) == out[i]:
                            correct += 1
                else:
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == out.long()).sum().item()
            
            predictions.extend(predicted.cpu().detach().numpy())
            pred_prob.extend(outputs.cpu().detach().numpy()[:,1])
            labels.extend(out.cpu().detach().numpy())
        train_loss_ls.append(running_loss/len(trainDataload))
        accuracy.append(100 * correct / total)
        print("Training Loss after epoch {} is {}".format(epoch + 1,
                                        running_loss / len(trainDataload),))
        val_loss,val_auc, val_prauc = validate(model,testDataload,criterion,epoch)
        if epochs > 5:
            if epoch == 0:
                best_val_loss = val_loss
            else:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = model    
                    torch.save(model.state_dict(), f"./checkpoints/{model_name}/{model_name}_e_{epoch + 1}.pth")            
                else:
                    pass
        else:
            best_model = model
            torch.save(model.state_dict(), f"./checkpoints/{model_name}/{model_name}_e_{epoch + 1}.pth")
        val_loss_ls.extend(val_loss)
    print(classification_report(np.array(labels),np.array(predictions)))
    validate(model,testDataload,criterion,epoch,True)
    model.feature_importances_ = torch.nn.Softmax(dim=0)(model.layers["0"].weight[1]).cpu().detach().numpy()
    return  best_model, train_loss_ls, val_loss_ls

@torch.no_grad()
def validate(model,testDataload,criterion,epoch,last=False):
    '''
    Function for validating the training on testing/validation data.
    :param model(TNN Classifier/Regressor): model to be trained
    :param testDataload(object of DataLoader): DataLoader with testing data
    :param criterion(object of loss function): Loss function to be used for training
    :param epoch(int,optional): Number of epochs for training the model. Default value: 100
    :param last(Boolean, optional): Checks if the current epoch is the last epoch. Default: False
    :return:
    list of validation loss,accuracy
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    auc_score, pr_auc_score = 0, 0
    valid_loss = 0
    accuracy, val_loss  = [], []
    pred_prob, predictions, labels= [], [], []
    correct = 0
    total = 0
    for i, data in enumerate(testDataload):
        ################
        inp, out = data[0].to(device), data[1].to(device)
        ################
        model.get(out.float())
        y_pred = model(inp.float(), train=False)
        if model.labels == 1:
            loss = criterion(y_pred, out.view(-1, 1).float())
        else:
            loss = criterion(y_pred, out.long())
        valid_loss += loss.item()
        total += out.float().size(0)
        predicted = y_pred
        if model.name == "Regression":
            pass
        else:
            if model.labels == 1:
                for i in range(len(y_pred)):
                    if y_pred[i] < torch.Tensor([0.5]):
                        y_pred[i] = 0
                    else:
                        y_pred[i] = 1
                    if y_pred[i].type(torch.LongTensor) == out[i]:
                        correct += 1
            else:
                _, predicted = torch.max(y_pred.data, 1)
                correct += (predicted == out.long()).sum().item()
        ########################
        predictions.extend(predicted.cpu().detach().numpy())
        pred_prob.extend(y_pred.cpu().detach().numpy()[:,1])
        labels.extend(out.cpu().detach().numpy())
    val_loss.append(valid_loss/ len(testDataload))
    accuracy.append(100 * correct / total)
    if last:
        print(classification_report(np.array(labels), np.array(predictions)))
    auc_score = roc_auc_score(labels, pred_prob)
    pr_auc_score = average_precision_score(labels, pred_prob)
    print("Validation Loss after epoch {} is {}".format(epoch + 1, valid_loss / len(testDataload)))
    return val_loss, auc_score, pr_auc_score

def predict(model,X):
    '''
    Predicts the output given the correct input data
    :param model(TNN Classifier/Regressor): model to be trained
    :param X: Feature for which prediction is required
    :return:
    predicted value(int)
    '''
    X = torch.from_numpy(X)
    y_pred = model(X.float(), train=False)
    if model.name == "Classification":
        if model.labels == 1:
            if y_pred < torch.Tensor([0.5]):
                y_pred = 0
            else:
                y_pred = 1
        else:
            y_pred = np.argmax(y_pred.cpu().detach().numpy(),axis=1)
        return y_pred
    else:
        return y_pred.cpu().detach().numpy()[0]

def predict_proba(model,X):
    '''
    Predicts the output given the correct input data
    :param model(TNN Classifier/Regressor): model to be trained
    :param X: Feature for which prediction is required
    :return:
    predicted probabilties value(int)
    '''
    X = torch.from_numpy(X)
    y_pred = model(X.float(), train=False)
    return y_pred