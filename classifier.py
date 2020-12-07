import numpy as np

import torch
import torchvision as tv
import torch.utils.data as td

import matplotlib.pyplot as plt

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
                
            
            
            

