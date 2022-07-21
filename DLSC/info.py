import numpy as np

class Info:
    def __init__(self):
        self.normtrigger = 0
        self.paralltrigger = 0
        self.parallcase = 'aver'
        
        self.solver = 'T'
        self.m = 64
        self.n = 256
        self.N = (int)(512-np.sqrt(self.m)+1)**2
        self.p = 1
        self.lamda = 1e-5
        self.alpha = 15
        self.sigma = 30
        self.maxLayer = 5
        self.eta=-1.2
        self.scale=1.8*1e-2