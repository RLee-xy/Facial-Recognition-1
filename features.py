import numpy as np

import torch.utils.data as td

from sklearn.decomposition import PCA
from skimage.feature import local_binary_pattern as LBP

##### PCA 
def pca_fit(dataset):
    # pca model
    pca = PCA()
    # dataloader
    dataloader = td.DataLoader(dataset, batch_size=dataset.__len__(), shuffle=False)
    for batch_idx, (X, Y) in enumerate(dataloader):
        print("Dimension of the batch data is", X.shape)
        X = X.squeeze().numpy()
        pca.fit(X)
    return pca

def pca_transform(pca, dataset):
    # dataloader
    dataloader = td.DataLoader(dataset, batch_size=dataset.__len__(), shuffle=False)
    for batch_idx, (X, Y) in enumerate(dataloader):
        print("Dimension of the batch data is", X.shape)
        X = X.squeeze().numpy()
        X_transformed = pca.transform(X)
    return X_transformed

##### LBP
def lbp_transform(dataset):
    # dataloader
    dataloader = td.DataLoader(dataset, batch_size=1, shuffle=False)
    X_transformed = []
    for batch_idx, (x, y) in enumerate(dataloader):
        x = x.squeeze().numpy()
        lbp = LBP(x, P=8*3, R=3, method='uniform')
        X_transformed.append(lbp)
    X_transformed = np.array(X_transformed)
    return X_transformed
        
