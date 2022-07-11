from unittest import FunctionTestCase
from cv2 import PSNR
import numpy as np
import matplotlib.pyplot as plt
import cv2

from function import Function
from Info import Info
from Record import Record

class WISTA:

    info = Info()
    record = Record()
    X0 = cv2.imread("AP_gray.png", 0).astype(float) #读取图片 512×512
    sigma = 20 #噪声强度
    X1 = Function.addNoise(X0, sigma, record)
    PSNR = Function.calculate_PSNR(X0,X1)
    X, D = Function.GeneData(X0, X1, info)
    lamda = 1e-5
    maxLayer = 100
    alpha = 15

    def forwardP(X, D, lamda, alpha, T):
        m, n = np.shape(D)
        M, N = np.shape(X)
        p = 0.7
        I = np.eye(n)
        DT = D.T
        eigenvalue, featurevector = np.linalg.eig(np.dot(DT,D))
        
        t = lamda/alpha
        z = np.zeros((n,N))
        H = I - np.dot(DT, D)/alpha
        W = DT / alpha
        B = np.dot(W,X)
        
        for k in range(0, T):
            # B = np.dot(D,z) - X
            # c = z - 1/alpha * np.dot(DT,B)
            # z = np.sign(c)*np.maximum(np.abs(c)-t, 0)
            # t = (lamda/alpha) * (np.abs(z)**(p-1))
            
            c = B + np.dot(H,z)
            z = np.sign(c)*np.maximum(np.abs(c)-t,0)
            t = lamda/alpha * np.abs(z)**(p-1)
            
            HP = Function.calculate_HoyerSparsity(z)
            print('iter:',k,'Hoyer Sparity:',HP)
           
            
        return z, D
    
    def backwardP(C, Z):
        return

    z, D = forwardP(X, D, lamda, alpha, maxLayer)
    X2 = np.dot(D,z)
    Img = Function.recoverImg(X2,512,8)
    PSNR1 = Function.calculate_PSNR(X0,Img)
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(X0, cmap='gray')
    plt.title('Origion image')
    plt.subplot(1, 3, 2)
    plt.imshow(X1, cmap='gray')
    plt.title('Noisy image, PSNR:%f' %PSNR)
    plt.subplot(1, 3, 3)
    plt.imshow(Img, cmap='gray')
    plt.title('Recover image, PSNR:%f' %PSNR1)
    plt.show()