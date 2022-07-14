import numpy as np
class Delta:
    
    def __init__(self, info):
        self.Z = np.empty((info.maxLayer+1, info.n, info.N))
        self.C = np.empty((info.maxLayer+1, info.n, info.N))
        self.t = np.empty((info.maxLayer+1, info.n, info.N))
        self.B = np.empty((info.maxLayer+1, info.n, info.N))
        self.H = np.empty((info.maxLayer+1, info.n, info.n))
        