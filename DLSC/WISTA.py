import numpy as np
import matplotlib.pyplot as plt
import cv2

from function import Function
from Info import Info
from Delta import Delta
from Record import Record

class WISTA:
    info = Info()
    record = Record(info)
    delta = Delta(info)
    m,n = 8,16
    X0 = cv2.imread("AP_gray.png", 0).astype(float) #读取图片 512×512
    X1, PSNR = Function.addNoise(X0, info.sigma)
    D = Function.geneDic(m,n)
    Xorig, X = Function.geneData(X0, X1, info)
    
    def wista(X, D, info):
        #init
        DT = D.T
        eigvs = np.linalg.eig(np.dot(DT,D))[0]
        alpha = 1.2*np.max(eigvs)
        lamda = 1e-5
        p = 0.7
        maxiter = 10
        n,N = info.n, info.N
        t = (lamda/alpha)*np.ones((n,N))
        z = np.zeros((n,N))
        k = 0
        e = 1e-15
        diff = 100
        for i in range(k, maxiter):
            temp = z - (1/alpha)*np.dot(DT,(np.dot(D,z)-X))
            z = Function.softTh(temp,t)[0]
            t = (lamda/alpha)*(np.abs(z)**(p-1))
            t[t==np.inf] = 0
            t[t==np.nan] = 0
            print('iter:',i)
        return z
    
    Z = wista(X, D, info)
    X2 = np.dot(D,Z)
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
            
        