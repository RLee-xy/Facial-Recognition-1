import numpy as np

import torch.utils.data as td

from sklearn.decomposition import PCA
from skimage.feature import local_binary_pattern as LBP
from skimage.feature import hog
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import FastICA

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

##### LDA 
def lda_fit(dataset, n_components):
    # lda model
    lda = LDA(n_components=n_components)
    # dataloader
    dataloader = td.DataLoader(dataset, batch_size=dataset.__len__(), shuffle=False)
    for batch_idx, (X, Y) in enumerate(dataloader):
        print("Dimension of the batch data is", X.shape)
        X = X.squeeze().numpy()
        Y = Y.numpy()
        lda.fit(X, Y)
    return lda

def lda_transform(lda, dataset):
    # dataloader
    dataloader = td.DataLoader(dataset, batch_size=dataset.__len__(), shuffle=False)
    for batch_idx, (X, Y) in enumerate(dataloader):
        print("Dimension of the batch data is", X.shape)
        X = X.squeeze().numpy()
        X_transformed = lda.transform(X)
    return X_transformed

##### LBP
def lbp_transform(dataset, cell=8):
    # dataloader
    dataloader = td.DataLoader(dataset, batch_size=1, shuffle=False)
    X_transformed = []
    hists = []
    for batch_idx, (x, y) in enumerate(dataloader):
        x = x.squeeze().numpy()
        lbp = LBP(x, P=8*3, R=3, method='uniform')
        X_transformed.append(lbp)
        n_bins = int(lbp.max() + 1)
        n_rows = lbp.shape[0] // cell
        n_cols = lbp.shape[1] // cell
        hist = []
        for i in range(n_rows):
            for j in range(n_cols):
                patch = lbp[i*cell:(i+1)*cell, j*cell:(j+1)*cell]
                h, _ = np.histogram(patch, density=True, bins=n_bins, range=(0, n_bins))
                hist.append(h)
        hist = np.concatenate(hist)
        hists.append(hist)
    X_transformed = np.array(X_transformed)
    hists = np.array(hists)
    return X_transformed, hists
        
#### HOG
def hog_transform(dataset):
    dataloader = td.DataLoader(dataset, batch_size=1, shuffle=False)
    fd_list = []
    hog_images = []

    for batch_idx, (X, Y) in enumerate(dataloader):
        X = X.squeeze().numpy()
        fd, hog_image = hog(X, orientations=9, pixels_per_cell=(5, 5),
                            cells_per_block=(2, 2), visualize=True, multichannel=False)
        fd_list.append(fd)
        hog_images.append(hog_image)

    return np.array(fd_list), hog_images

##### ICA
def ica_fit(dataset, n_component=50):
    # ica model
    ica = FastICA(n_components=n_component)
    flag = False
    # dataloader
    dataloader = td.DataLoader(dataset, batch_size=dataset.__len__(), shuffle=False)
    for batch_idx, (X, Y) in enumerate(dataloader):
        print("Dimension of the batch data is", X.shape)
        X = X.squeeze().numpy()
        try:
            X_train = ica.fit_transform(X)
            flag = True
        except:
            flag = False
            pass
    return ica, flag

def ica_transform(ica, dataset):
    # dataloader
    dataloader = td.DataLoader(dataset, batch_size=dataset.__len__(), shuffle=False)
    for batch_idx, (X, Y) in enumerate(dataloader):
        print("Dimension of the batch data is", X.shape)
        X = X.squeeze().numpy()
        X_transformed = ica.transform(X)
    return X_transformed
