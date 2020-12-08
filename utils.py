import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

import torch

from sklearn.manifold import TSNE


def save_model(model, path):
    folder = os.path.split(path)[0]
    if folder != '':
        os.makedirs(folder, exist_ok=True)
    if isinstance(model, torch.nn.Module):
        torch.save(model, path)
    else:
        with open(path, 'wb') as file:
            pickle.dump(model, file)
    print("Model saved to %s." % path)


def load_model(path):
    basename = os.path.basename(path)
    _, ext = os.path.splitext(basename)
    if ext == ".pth":
        return torch.load(path)
    elif ext == ".pkl":
        with open(path, 'rb') as file:
            return pickle.load(file)
    else:
        raise ValueError("Invalid file format.")

def visualize_tsne(X, Y):
    X = np.array(X)
    Y = np.array(Y)
    tsne = TSNE(n_components=2)
    tsne_results = tsne.fit_transform(X)
    plt.figure(figsize=(12, 8))
    plt.title("t-SNE visualization")
    for i in range(38):
        plt.scatter(x=tsne_results[Y==i, 0], y=tsne_results[Y==i, 1], \
                    alpha=0.3, label=i)
    plt.legend()
    plt.show()

    
                
            
            
            

