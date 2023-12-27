import numpy as np
from numpy.random import *

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
        # A:np.ndarray = rand(m, n) * 2 - 1
        A: np.ndarray = np.zeros((m, n))
        for i in range(m):
            A[i] = np.random.normal(0, 0, n)
        return A.round(0).astype(int)
    
    def genPk(self):
        A = self.RandArray(self.n, self.n, self.q)
        S = self.RandArray(self.n, 1, self.q)
        e = self.errorArray(self.n)
        T = np.matmul(A, S) + e
        return A, T%self.q, S

    def enc(self, A:np.ndarray, T:np.ndarray, m:np.ndarray):
        r = self.RandArray(self.n, 1, self.q)
        e1 = self.errorArray(self.n)
        e2 = self.errorArray(1)
        U = np.matmul(A.transpose(), r) + e1
        U = U  % self.q
        C = np.matmul(T.transpose(), r) + e2 + m*np.round(self.q/2)
        C = C % self.q
        return U, C

    def dec(self, U:np.ndarray, C:np.ndarray, S:np.ndarray):
        m = C - np.matmul(S.transpose(), U)
        m = m %self.q
        return 1 if (m%self.q) < 3*self.q//4 and (m%self.q) > self.q//4 else 0

if __name__ == "__main__":
    # seed(5)
    q = 13
    n = 7
    n = int(input("n to be :"))
    q = int(input("q to be :"))
    msg_len = int(input("message bits length :"))
    msg_bit = randint(0,2,msg_len) # random bit
    # a encryption machine to carry out the implemetation
    LWE_test = LWE(n,q) 

    # generate a pair of public key and secret key
    A, T, S= LWE_test.genPk()
    # an eavesdropper is listening
    eve_S = np.matmul(np.linalg.inv(A), T).round(0).astype(int) % q
    eve_bits = []

    msg_dec = np.zeros(msg_len)
    for i in range(msg_len):
        # encodeing msg
        U,C = LWE_test.enc(A,T,msg_bit[i]) 
        # decoding the ciphertext
        msg_dec[i] = LWE_test.dec(U,C,S) 
        
        # an eavesdropper is listening
        eve_bits.append(LWE_test.dec(U, C, eve_S))
    
    # prompt the result
    print("-----pk and sk:-----")
    print(f'A=\n{A}')    
    print(f'T=\n{T}')
    
    print(f'S=\n{S}')
    print(f'eve_S=\n{eve_S}')

    print("-----ciphertext:-----")
    print(f'U=\n{U}')
    print(f'C=\n{C}')

    # eve_bits = [0 if (C_list[i]%q) < 3*q//4 and (C_list[i]%q) > q//4 else 1 for i in range(len(C_list))]
    print("-----result:-----")
    print("message bit:", msg_bit)
    print("eve_bits:", np.array(eve_bits))
    print("decoding bit:", msg_dec.astype(int))
    # if all(e1 == e2 for e1,e2 in zip(msg_bit, msg_dec)) :
    #     print("\nmsg_bit == msg_dec")
    # else : 
    #     print ("\nWrong!!! msg_bit != msg_dec ")
    print("Decode BER",np.mean(msg_bit != msg_dec))
    print("eve BER", np.mean(msg_bit != eve_bits))