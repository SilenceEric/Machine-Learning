import numpy as np
import matplotlib.pyplot as plt
import cv2

from function import Function
from Info import Info
from Delta import Delta
from Record import Record

class DNNWISTA:
    maxIter = 1
    iterr = 5
    m,n = 8,16
    info = Info()
    record = Record(info)
    delta = Delta(info)
    
    X0 = cv2.imread("AP_gray.png", 0).astype(float) #读取图片 512×512
    X1, PSNR = Function.addNoise(X0, info.sigma)
    Dic = Function.geneDic(m,n)
    Xorig, X, D = Function.geneData(X0, X1, Dic, info)
    
    record.DT = D.T
    
    def forwardP(X, D, info, record, T):
        n = info.n
        N = info.N
        p = info.p
        I = np.eye(n)
        DT = record.DT
        lamda = info.lamda
        alpha = info.alpha
        eigenvalue, featurevector = np.linalg.eig(np.dot(DT,D))
        
        Z = C = np.empty((T,n,N))
        t = np.ones((T,n,N))
        H = I - np.dot(DT, D)/alpha + info.eta*info.scale*record.dH
        W = DT / alpha + info.eta*info.scale*record.dW
        B = np.dot(W,X)
        t[0,:,:] = t[0,:,:]*(lamda/alpha) + info.eta*info.scale*record.dt
        
        for k in range(1, T):
            # B = np.dot(D,z) - X
            # c = z - 1/alpha * np.dot(DT,B)
            # z = np.sign(c)*np.maximum(np.abs(c)-t, 0)
            # t = (lamda/alpha) * (np.abs(z)**(p-1))
            
            C[k-1,:,:] = B + np.dot(H,Z[k-1,:,:])
            Z[k,:,:] = Function.softTh(C[k-1,:,:], t[k-1,:,:])[0]
            t[k,:,:] = lamda/alpha * (np.abs(Z[k,:,:])**(p-1))
            t[k,:,:][t[k,:,:] == np.inf] = 0
            
            HP = Function.calculate_HoyerSparsity(Z[k,:,:])
            print('iter:',k,'Hoyer Sparity:',HP)
        
        record.t = t
        record.HT = H.T
        return Z, C
    
    def backwardP(x, D, Z, C, delta, info, record):
        T = info.maxLayer
        lamda = info.lamda
        DT = record.DT
        HT = record.HT
        t = record.t
        
        if info.solver == 'L':
            delta.Z[T,:,:] = Z[T,:,:] - record.Z
        elif info.solver == 'T':
            delta.Z[T,:,:] = np.dot(DT,(np.dot(D, Z[T,:,:]) - x)) + lamda * np.sign(Z[T,:,:])
        else: delta.Z[T,:,:] = delta.Z[T,:,:]

            
        for k in range(T-1,-1,-1):
            delta.t[k,:,:] = Function.softTh(C[k,:,:],t[k,:,:])[1]*delta.Z[k+1,:,:]
            delta.C[k,:,:] = Function.softTh(C[k,:,:],t[k,:,:])[2]*delta.Z[k+1,:,:]
            delta.B[k,:,:] = delta.B[k+1,:,:] + delta.C[k,:,:]
            delta.H[k,:,:] = delta.H[k+1,:,:] + np.dot(delta.C[k,:,:], Z[k,:,:].T)
            delta.Z[k,:,:] = np.dot(HT, delta.C[k,:,:]) + Function.updateZ(Z[k,:,:], info)*delta.t[k,:,:]
            
        dW = np.dot(delta.B[0,:,:],x.T)
        dH = delta.H[0,:,:]
        dt = delta.t[0,:,:]
        return dW, dH, dt
    
    Z, C = forwardP(X, D, info, record, info.maxLayer+1)
    record.Z = Z[info.maxLayer,:,:]
    for i in range(0, maxIter):
        record.dW,record.dH,record.dt = backwardP(X, D, Z, C, delta, info, record)
        Z, C = forwardP(X, D, info, record, info.maxLayer+1)
    
    X2 = np.dot(D,Z[iterr,:,:])
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