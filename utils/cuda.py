import torch

def cudarize(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor