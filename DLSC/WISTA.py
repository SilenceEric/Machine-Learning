import numpy as np
import matplotlib.pyplot as plt
import cv2

from function import Function
from Info import Info
from Delta import Delta
from Record import Record

class WISTA:
    info = Info()
    record = Record()
    delta = Delta()
    
    X0 = cv2.imread("AP_gray.png", 0).astype(float) #读取图片 512×512
    X1 = Function.addNoise(X0, info.sigma, record)
    PSNR = Function.calculate_PSNR(X0,X1)
    X, D = Function.GeneData(X0, X1, info)

    def forwardP(X, D, info):
        n = np.size(D,1)
        N = np.size(X,1)
        p = 0.7
        I = np.eye(n)
        DT = D.T
        lamda = info.lamda
        alpha = info.alpha
        T = info.maxLayer+1
        eigenvalue, featurevector = np.linalg.eig(np.dot(DT,D))
        
        t = lamda/alpha * np.ones((T,n,N))
        Z = np.zeros((T,n,N))
        C = np.zeros((T,n,N))
        H = I - np.dot(DT, D)/alpha
        W = DT / alpha
        B = np.dot(W,X)
        
        for k in range(1, T):
            # B = np.dot(D,z) - X
            # c = z - 1/alpha * np.dot(DT,B)
            # z = np.sign(c)*np.maximum(np.abs(c)-t, 0)
            # t = (lamda/alpha) * (np.abs(z)**(p-1))
            
            C[k-1,:,:] = B + np.dot(H,Z[k-1,:,:])
            Z[k,:,:] = Function.softh(C[k-1,:,:], t[k-1,:,:])
            t[k,:,:] = lamda/alpha * np.abs(Z[k,:,:])**(p-1)
            
            HP = Function.calculate_HoyerSparsity(Z[k,:,:])
            print('iter:',k,'Hoyer Sparity:',HP)
            
        return Z, C
    
    def backwardP(x, D, Z, C, delta, info):
        T = info.maxLayer
        lamda = info.lamda
        alpha = info.alpha
        DT = D.T
        HT = info.H.T
        
        if info.solver == 'L':
            delta.Z[:,:,T] = Z[:,:,T] - info.Z
        elif info.solver == 'T':
            delta.Z[:,:,T] = np.dot(DT,(np.dot(D,Z[:,:,T]) - x)) + lamda * np.sign(Z[:,:,T])
            
        for k in range(T-1,0):
            delta.t[:,:,k] = delta.Z[:,:,k+1]
        return

    Z,C = forwardP(X, D, info)
    X2 = np.dot(D,Z[info.maxLayer,:,:])
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