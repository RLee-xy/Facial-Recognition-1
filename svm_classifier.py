import sklearn
import features
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

class SVM_agent:
    def __init__(self):
        self.agent = sklearn.svm.LinearSVC(penalty='l2', max_iter=5000, random_state=0)
        self.train_data = []
        self.train_label = []
        self.test_data = []
        self.test_label = []
        self.params_search = {'C':[1,10,100,1000]}


    def reset(self):
        self.agent = sklearn.svm.LinearSVC()
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
        print("Accuracy:" + str(accuracy_score(self.test_label, pred_labels)))

    def fit(self):
        self.agent.fit(self.train_data, self.train_label)

    def pred(self):
        pred_labels = self.agent.predict(self.test_data)
        return pred_labels

    def hyper_tune(self, dataset, labels):
        self.agent = sklearn.svm.LinearSVC()
        if len(dataset.shape) == 3:
            self.train_data = np.array(dataset).reshape((dataset.shape[0], dataset.shape[1] * dataset.shape[2]))
        else:
            self.train_data = np.array(dataset)
        self.train_label = np.array(labels)

        hp_search = GridSearchCV(estimator=self.agent, param_grid=self.params_search, n_jobs=-1, cv=3)
        hp_search.fit(self.train_data, self.train_label)
        self.agent = hp_search.best_estimator_
        print("Best parameters:", hp_search.best_params_)


