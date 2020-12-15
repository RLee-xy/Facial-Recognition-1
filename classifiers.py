import numpy as np

import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC

from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

import torch
torch.manual_seed(42)
import torchvision as tv
import torch.utils.data as td

import matplotlib.pyplot as plt


##### KNN
def train_knn(X_train, Y_train):
    model = KNeighborsClassifier()
    param_grid = {'n_neighbors': [3, 5, 10], 'weights': ['uniform', 'distance'], 'p': [1, 2]}
    hp_search = GridSearchCV(estimator=model, param_grid=param_grid)
    hp_search.fit(X_train, Y_train)
    best_model = hp_search.best_estimator_
    print("Best parameters:", hp_search.best_params_)
    return best_model


##### SVM
class SVM_agent:
    def __init__(self):
        self.agent = SVC(random_state=42)
        self.train_data = []
        self.train_label = []
        self.test_data = []
        self.test_label = []
        self.params_search = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}


    def reset(self):
        self.agent = SVC(random_state=42)
        self.train_data = []
        self.train_label = []
        self.test_data = []
        self.test_label = []


    def train(self, dataset, labels):
        if len(dataset.shape) == 3:
            self.train_data = np.array(dataset).reshape((dataset.shape[0], dataset.shape[1] * dataset.shape[2]))
        else:
            self.train_data = np.array(dataset)
        self.train_label = np.array(labels)
        self.fit()

    def test(self, dataset, labels):
        if len(dataset.shape) == 3:
            self.test_data = np.array(dataset).reshape((dataset.shape[0], dataset.shape[1] * dataset.shape[2]))
        else:
            self.test_data = np.array(dataset)
        self.test_label = np.array(labels)
        pred_labels = self.pred()
        acc = accuracy_score(self.test_label, pred_labels)
        return acc

    def fit(self):
        self.agent.fit(self.train_data, self.train_label)

    def pred(self):
        pred_labels = self.agent.predict(self.test_data)
        return pred_labels

    def hyper_tune(self, dataset, labels):
        self.agent = SVC(random_state=42)
        if len(dataset.shape) == 3:
            self.train_data = np.array(dataset).reshape((dataset.shape[0], dataset.shape[1] * dataset.shape[2]))
        else:
            self.train_data = np.array(dataset)
        self.train_label = np.array(labels)

        hp_search = GridSearchCV(estimator=self.agent, param_grid=self.params_search, n_jobs=-1, cv=3)
        hp_search.fit(self.train_data, self.train_label)
        self.agent = hp_search.best_estimator_
        print("Best parameters:", hp_search.best_params_)

    
##### ResNet
def train_resnet(trainset, testset, batch_size=128, epochs=100, lr=0.001, show_progress=True):
    # dataloader
    train_loader = td.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = td.DataLoader(testset, batch_size=testset.__len__(), shuffle=False)
    # model
    model = tv.models.resnet18(pretrained=False)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = torch.nn.Linear(in_features=512, out_features=38, bias=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    # optimizer
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=500, gamma=0.1)
    # statistics
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    for e in range(epochs):
        model.train()
        running_loss = 0
        n_samples = 0
        correct = 0
        for _, (X_train, Y_train) in enumerate(train_loader):
            X_train = X_train.to(device)
            Y_train = Y_train.to(device)
            batch_size = X_train.shape[0]
            Y_pred = model(X_train)
            loss = torch.nn.functional.cross_entropy(Y_pred, Y_train)
            opt.zero_grad()
            loss.backward()
            opt.step()
            lr_scheduler.step()
            with torch.no_grad():
                running_loss += loss.item() * batch_size
                n_samples += batch_size
                _, preds = Y_pred.max(-1)
                correct += (preds == Y_train).sum().float()
                
        with torch.no_grad():
            train_loss.append(running_loss / n_samples)
            train_acc.append(correct / n_samples)
            # evaluate
            model.eval()
            X_test, Y_test = next(iter(test_loader))
            X_test = X_test.to(device)
            Y_test = Y_test.to(device)
            Y_pred = model(X_test)
            loss = torch.nn.functional.cross_entropy(Y_pred, Y_test)
            _, preds = Y_pred.max(-1)
            correct = (preds == Y_test).sum().float()
            test_loss.append(loss.item())
            test_acc.append(correct / X_test.shape[0])

        if e % 10 == 9:
            print("Epoch: %2d | Train Loss: %.3f | Train Acc: %.2f%% | Test Loss: %.3f | Test Acc: %.2f%%" \
                  % (e+1, train_loss[e], train_acc[e]*100, test_loss[e], test_acc[e]*100))
    # plot
    if show_progress:
        plt.plot(np.arange(epochs), train_loss, label="train")
        plt.plot(np.arange(epochs), test_loss, label="test")
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.legend()
        plt.show()
        plt.plot(np.arange(epochs), train_acc, label="train")
        plt.plot(np.arange(epochs), test_acc, label="test")
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.legend()
        plt.show()
        
    return model
                
            
            
            

