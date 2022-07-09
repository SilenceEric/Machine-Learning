import numpy as np
import matplotlib.pyplot as plt
import geneData

from Info import Info
from Record import Record

info = Info()
Record = Record()
[X, D] = geneData.GeneData(info, Record)
lamda = 200e-0
alpha = 100
maxLayer = 5

def forwardP(X, D, lamda, alpha, T):
    [m, n] = np.shape(D)
    [M, N] = np.shape(X)
    p = 2
    I = np.eye(n)
    DT = D.T
    
    t = lamda/alpha
    z = np.zeros((n,N))
    H = I - np.dot(DT, D)/alpha
    W = DT / alpha
    B = np.dot(W,X)
    
    for k in range(1, T):
        c = B + np.dot(H,z)
        z = np.sign(c)*np.maximum(np.abs(c)-t,0)
        t = lamda/alpha * np.abs(z)**(p-1)
        
    return z
    
def backwardP():
    return

z = forwardP(X, D, lamda, alpha, maxLayer)
X1 = np.dot(D,z)
for i in range(0,np.size(X,1)):
    X[i].reshape()