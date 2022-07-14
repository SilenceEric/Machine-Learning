import numpy as np
class Record:
    def __init__(self, info):
        self.PSNRinpuut = 0
        self.t = np.empty((info.maxLayer+1, info.n, info.N))
        self.HT =None
        self.DT = None
        self.Z = None
        self.dW = np.zeros((info.n, info.m))
        self.dH = np.zeros((info.n, info.n))
        self.dt = np.zeros((info.n, info.N))