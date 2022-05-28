import numpy as np
import matplotlib.pyplot as plt

np.random.seed()
N = 100
K = 3
T3 = np.zeros((N,3), dtype=np.uint8)
X = np.zeros((N,2))
X_range0 = [-3, 3]
X_range1 = [-3, 3]
x_col = ['cornflowerblue', 'black', 'white']
Mu = np.array([[-.5, -.5],[.5, 1.0], [1,-.5]])
Sig = np.array([[.7, .7],[.8, .3],[.3, .8]])
Pi = np.array([0.4, 0.8, 1])
for n in range(N):
    wk = np.random.rand()
    for k in range(K):
        if wk < Pi[k]:
            T3[n, k] = 1
            break
    for k in range(2):
        X[n, k] = (np.random.randn() * Sig[T3[n,:] == 1, k] + Mu[T3[n, :] == 1, k])
        
def show_data(x):
    plt.plot(x[:, 0], x[:, 1], linestyle='none', marker='o', markersize=6, markeredgecolor='black', color='gray', alpha=0.8)
    plt.grid(True)

plt.figure(1, figsize=(4, 4))
show_data(X)
plt.xlim(X_range0)
plt.ylim(X_range1)
plt.show()
np.savez('data_ch9.npz',X=X, X_range0=X_range0, X_range1=X_range1)

Mu = np.array([[-2, 1],[-2, 0], [-2, -1]])
R = np.c_[np.ones((N, 1), dtype=int), np.zeros((N,2), dtype=int)]

def show_prm(x, r, mu, col):
    for k in range(K):
        plt.plot(x[r[:, k] == 1, 0], x[r[:,k] == 1, 1],marker='o', markerfacecolor=x_col[k], markeredgecolor='k', markersize=6, alpha=0.5, linestyle='none')
        
        plt.plot(mu[k, 0], mu[k, 1], marker='*', markerfacecolor=x_col[k], markersize=15,markeredgecolor='k',markeredgewidth=1)
    
    plt.xlim(X_range0)
    plt.ylim(X_range1)
    plt.grid(True)
    
plt.figure(figsize=(4, 4))
R = np.c_[np.ones((N, 1)), np.zeros((N, 2))]
show_prm(X, R, Mu, x_col)
plt.title("initial Mu and R")
plt.show()

def step1_kmeans(x0, x1, mu):
    N = len(x0)
    r = np.zeros((N, K))
    for n in range(N):
        wk = np.zeros(K)
        for k in range(K):
            wk[k] = (x0[n] - mu[k, 0])**2 +(x1[n] - mu[k, 1])**2
        r[n, np.argmin(wk)] = 1
    return r

plt.figure(figsize=(4,4))
R = step1_kmeans(X[:, 0], X[:, 1], Mu)
show_prm(X, R, Mu, x_col)
plt.title('Step 1')
plt.show()

def step2_kmeans(x0, x1, r):
    mu = np.zeros((K, 2))
    for k in range(K):
        mu[k, 0] = np.sum(r[:, k] * x0) / np.sum(r[:, k])
        mu[k, 1] = np.sum(r[:, k] * x1) / np.sum(r[:, k])
    return mu

plt.figure(figsize=(4,4))
Mu = step2_kmeans(X[:, 0], X[:, 1], R)
show_prm(X, R, Mu, x_col)
plt.title('Step2')
plt.show()

plt.figure(1, figsize=(10,6.5))
Mu = np.array([[-2, 1], [-2, 0],[-2,-1]])
max_it = 6

for it in range(0, max_it):
    plt.subplot(2, 3, it+1)
    R = step1_kmeans(X[:, 0], X[:, 1], Mu)
    show_prm(X, R, Mu, x_col)
    plt.title("{0:d}".format(it + 1))
    plt.xticks(range(X_range0[0], X_range0[1]) ,"")
    plt.yticks(range(X_range1[0], X_range1[1]) ,"")
    Mu = step2_kmeans(X[:, 0], X[:,1], R)
plt.show()