# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import cv2

from Info import Info
from Record import Record

class Function:
    def addNoise(y0, sigma):
        N = np.size(y0,0) #512
        noise = abs(np.random.randn(N, N)) #随机生成高斯噪声
        y1 = y0 + sigma * noise #添加噪声
        PSNR = 10 * np.log10(255**2 / np.mean(np.power((y1[:] - y0[:]), 2))) #计算PSNR峰值信噪比
        return y1, PSNR
    
    def geneDic(m, n):
        Dictionary = np.zeros((m,n))
        for k in range(0, n):
            V = np.cos(np.arange(0,m)*np.pi*k/n)
            if k > 0:
                V = V - np.mean(V)
            Dictionary[:, k] = V/np.linalg.norm(V)

        Dictionary = np.kron(Dictionary, Dictionary)
        Dictionary = np.dot(Dictionary, np.diag(1/np.sqrt(np.sum(Dictionary * Dictionary, 0))))
        return Dictionary
    
    def geneData(y0, y1, info):
        m = 8
        # NoTotal = np.size(y0, 0)-m+1
        # cnt, Kdata = 0,1
        # sidenum = int(np.ceil((NoTotal-1)/Kdata)+1)
        # Data = np.zeros((m**2, sidenum**2))
        # Datanorm = np.ones((1, sidenum**2))

        # if np.mod((NoTotal-1), Kdata) == 0:
        #     for j in range(0, NoTotal, Kdata):
        #         for i in range(0, NoTotal, Kdata):
        #             patch = y1[i:i+m,j:j+m].reshape(m**2)
        #             Data[:, cnt] = patch[:]
        #             cnt = cnt + 1
        # else:
        #     i,j = 0
        #     for k in range(0,sidenum**2):
        #         patch = y1[i:i+m, j:j+m].reshape(m**2)
        #         Data[:, k] = patch[:]
        #         if i<np.size(y1,0)-2*m:
        #             i = i+Kdata
        #         elif i<np.size(y1,0)-m:
        #             i = np.size(y1,0) - m
        #         else:
        #             i = 1
        #             if j<np.size(y1,0)-2*m:
        #                 j=j+Kdata
        #             else: 
        #                 j=np.size(y1,0)-m

        # if info.paralltrigger:
        #     if info.parallcase == 'aver':
        #         Data = Data - np.sum(Data) / (m**2*np.size(Data,1))

        # if info.normtrigger:
        #     for i in range(0, np.size(Data, 1)):
        #         Datanorm[:,i] = np.linalg.norm(Data[:,i])
        #         Data[:,i] = patch[:]/Datanorm[:,i]
        
        NoTotal = np.size(y0,0)-m+1
        cnt, Kdata1 = 0,1
        Data = np.zeros((m**2,int((NoTotal-1)/Kdata1+1)**2))
        Datanorm = np.zeros((1,int((NoTotal-1)/Kdata1+1)**2))
        for j in range(0, NoTotal, Kdata1):
            for i in range(0, NoTotal, Kdata1):
                patch = y0[i:i+m,j:j+m].reshape(m**2)
                Data[:,cnt] = patch[:]
                #DataZ(:,:,cnt)=patch(:,:);
                cnt += 1

        if info.paralltrigger:
            if info.parallcase == 'aver':
                Data = Data - np.sum(Data) / (m**2*np.size(Data,1))

        if info.normtrigger:
            for i in range(0, np.size(Data, 1)):
                Datanorm[:,i] = np.linalg.norm(Data[:,i])
                Data[:,i] = patch[:]/Datanorm[:,i]

        NoTotal = np.size(y1,0)-m+1
        cnt, Kdata1 = 0,1
        Data1 = np.zeros((m**2,int((NoTotal-1)/Kdata1+1)**2))
        Data1norm = np.zeros((1,int((NoTotal-1)/Kdata1+1)**2))
        for j in range(0, NoTotal, Kdata1):
            for i in range(0, NoTotal, Kdata1):
                patch = y1[i:i+m,j:j+m].reshape(m**2)
                Data1[:,cnt] = patch[:]
                #DataZ(:,:,cnt)=patch(:,:);
                cnt += 1

        if info.paralltrigger:
            if info.parallcase == 'aver':
                Data1 = Data1 - np.sum(Data1) / (m**2*np.size(Data1,1))

        if info.normtrigger:
            for i in range(0, np.size(Data1, 1)):
                Data1norm[:,i] = np.linalg.norm(Data1[:,i])
                Data1[:,i] = patch[:]/Data1norm[:,i]

        Record.PSNRinpuut = 10 * np.log10(255**2 / np.mean(np.power((y1[:] - y0[:]), 2)))
        
        return Data, Data1

    def recoverImg(Data, imgSize, k):
        img = np.zeros((imgSize,imgSize))
        cnt = 0
        t = imgSize-k+1
        for j in range(0, t, 1):
            for i in range(0, t, 1):
                patch = Data[:,cnt].reshape(k,k)
                img[i:i+k, j:j+k] = patch
                cnt += 1
        return img
    
    def calculate_PSNR(X0, X1):
        PSNR = 10 * np.log10(255**2 / np.mean(np.power((X1[:] - X0[:]), 2)))
        return PSNR
    
    def calculate_HoyerSparsity(Z):
        sqrtn = np.sqrt(np.size(Z))
        norm = np.sum(np.abs(Z)) / np.sqrt(np.sum(Z**2))
        HP = (sqrtn-norm) / (sqrtn-1)
        return HP
    
    def softTh(x, t):
        px = abs(x) - t
        px[px>0] = 1
        px[px<=0] = 0
        rt = (x-t*x/np.abs(x))*px
        rt[rt == np.inf] = 0
        pt = np.abs(rt)
        return rt, px, pt
    
    def updateZ(zk, info):
        z2 = np.abs(zk)**(info.p-2)
        signz = np.sign(zk)
        rt = info.lamda*(info.p-1)*z2*signz / info.alpha
        return rt
        

    if __name__ == "__main__":
        info = Info()
        record = Record(info)
        X0 = cv2.imread("AP_gray.png", 0).astype(float) #读取图片 512×512
        X1 = addNoise(X0, info.sigma, record)
        Xorig, X, D = geneData(X0, X1, info)
        #Img = recoverImg(Data,512,8)
        
        plt.imshow(D, cmap="gray")
        plt.title("DCT Dictionary")
        plt.show()