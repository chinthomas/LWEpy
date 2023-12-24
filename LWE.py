import numpy as np
from numpy.random import *
import sys

class LWE:
    def __init__(self, n, q) -> None:
        """
            determine some parameters for the cryto
        """
        self.n = n
        self.q = q
    
    @staticmethod
    def RandArray(n:int, m:int, q:int)-> np.ndarray:
        """
            generate a random number array with size n*m
        """
        A:np.ndarray = rand(n, m) * q
        A = A.round(0).astype(int)
        return A
    
    @staticmethod
    def errorArray(m, n=1):
        # make error is random number of set [-1 0 1]
        A:np.ndarray = rand(m, n) * 2 - 1
        return A.round(0).astype(int)
    
    def genPk(self):
        A = self.RandArray(self.n, self.n, self.q)
        S = self.RandArray(self.n, 1, self.q)
        e = self.errorArray(self.n)
        T = np.matmul(A, S) + e
        return A, T%q, S


    def enc(self, A:np.ndarray, T:np.ndarray, m):
        r = self.RandArray(self.n, 1, self.q)
        e1 = self.errorArray(self.n)
        e2 = self.errorArray(1)
        U = np.matmul(A.transpose(), r) % self.q + e1
        C = np.matmul(T.transpose(), r) + e2 + m*(self.q//2)
        C = C % q
        return U, C

    def dec(self, U:np.ndarray, C:np.ndarray, S:np.ndarray):
        m = C - np.matmul(S.transpose(), U)
        return 0 if (m%self.q) > self.q//2 else 1

if __name__ == "__main__":
    seed(5)
    q = 13
    n = 7
    LWE_test = LWE(n,q)
    A, T, S= LWE_test.genPk()
    print(f'A=\n{A}')    
    print(f'T=\n{T}')
    print(f'S=\n{S}')
    m1 = 0
    U,C = LWE_test.enc(A,T,m1)
    print(f'U=\n{U}')
    print(f'C=\n{C}')
    m2 = LWE_test.dec(U,C,S)
    if m1 == m2 :
        print("m1 == m2")
    else : 
        print ("Wrong!!! m1 != m2 ")