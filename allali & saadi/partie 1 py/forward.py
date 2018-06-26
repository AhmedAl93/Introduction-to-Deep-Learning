import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def forward(batch, W, b):
    s=batch*W+b
    e_x = np.exp(s - np.max(s))
    soft = e_x / e_x.sum(axis=0)
    return soft
    
    
    
    
    
    
    
    