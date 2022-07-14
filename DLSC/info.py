import numpy as np

class Info:
    def __init__(self):
        self.normtrigger = 0
        self.paralltrigger = 0
        self.parallcase = 'aver'
        
        self.solver = 'L'
        self.m = 8**2
        self.n = 16**2
        self.N = (int)(512-np.sqrt(self.m)+1)**2
        self.p = 2
        self.lamda = 1e-5
        self.alpha = 15
        self.sigma = 20
        self.maxLayer = 5