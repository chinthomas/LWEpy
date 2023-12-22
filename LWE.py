import numpy as np
from numpy.random import *
import sys

def genPk(n, q):
    A = RandArray(n, n, q)
    S = RandArray(n, 1, q)
    e = errorArray(n)
    T = np.matmul(A, S) + e
    return A, T%q, S

def RandArray(n, m, q)-> np.ndarray:
    A:np.ndarray = rand(n, m) * q
    A = A.round(0).astype(int)
    return A

def errorArray(m, n=1):
    # make error is random number of set [-1 0 1]
    A:np.ndarray = rand(m, n) * 2 - 1
    return A.round(0).astype(int)

def enc(A:np.ndarray, T:np.ndarray, m, n, q):
    r = RandArray(n, 1, q)
    e1 = errorArray(n)
    e2 = errorArray(1)
    U = np.matmul(A.transpose(), r) % q + e1
    C = np.matmul(T.transpose(), r) + e2 + m*(q//2)
    C = C % q
    return U, C

def dec(U:np.ndarray, C:np.ndarray, S:np.ndarray, n, q):
    m = C - np.matmul(S.transpose(), U)
    return 0 if (m%q) > q//2 else 1

if __name__ == "__main__":
    q = 13
    n = 7
    m1 = 0
    seed(5)
    A, T, S= genPk(n,q)
    print(f'A=\n{A}')    
    print(f'T=\n{T}')
    print(f'S=\n{S}')
    U,C = enc(A,T,m1,n,q)
    print(f'U=\n{U}')
    print(f'C=\n{C}')
    m2 = dec(U,C,S,n,q)
    if m1 == m2 :
        print("m1 == m2")
    else : 
        print ("Wrong!!! m1 != m2 ")