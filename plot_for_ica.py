import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.utils.data as td

from dataset import ExtendedYaleFace
from features import *
from classifiers import *
from utils import *
import time
import json
import pandas as pd
import tqdm


# train_set = ExtendedYaleFace(root="data/CroppedYale", image_shape=[32, 32], flatten=True, normalize=True, train=True)
# test_set = ExtendedYaleFace(root="data/CroppedYale", image_shape=[32, 32], flatten=True, normalize=True, test=True)
#
# svm_agent = SVM_agent()
#
# time_list = []
# accuracy_list = []
#
# for i in tqdm.tqdm(range(1,1024,10)):
#     svm_agent.reset()
#     start = time.time()
#     ica, flag =ica_fit(train_set, n_component=i)
#     if flag:
#         ica_train = ica_transform(ica, train_set)
#         ica_test = ica_transform(ica, test_set)
#         svm_agent.train(ica_train, train_set.labels)
#         accuracy = svm_agent.test(ica_test, test_set.labels)
#         end = time.time()
#         time_list.append(start-end)
#         accuracy_list.append(accuracy)
#
# df_time = pd.DataFrame({'time':time_list, 'accuracy':accuracy_list})
# df_time.to_csv('time.csv', index=False, sep=',')
#
# plt.figure()
# plt.plot(range(1,1024,1), accuracy_list)
# plt.xlabel('number of component')
# plt.ylabel('accuracy')
#
# plt.figure()
# plt.plot(range(1,1024,1), time_list)
# plt.xlabel('number of component')
# plt.ylabel('time cost')

df = pd.read_csv('time.csv')
plt.figure(figsize=(10,6))
plt.plot(np.arange(len(df['accuracy']))*10, df['accuracy'])
plt.xlabel('Number of Components')
plt.ylabel('Accuracy')
plt.xlim(-50, 1040)
plt.ylim(0.0, 1)
plt.grid(axis='x')
locs, _ = plt.xticks(np.arange(-50,1050,50 ))
locs, _ = plt.yticks(np.arange(0.0, 1, 0.05 ))
plt.title('Parameter for ICA')
plt.show()