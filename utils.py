import torch

def project(x, norm):
    if torch.norm(x, p=1) >= norm:
        return x / torch.norm(x, p=1) * norm
    else:
        return x

