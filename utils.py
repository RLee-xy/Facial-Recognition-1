import os
import numpy as np
import pickle

import torch


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

    
                
            
            
            

