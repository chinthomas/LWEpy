from numpy.polynomial import Polynomial
from numpy.random import*
import numpy as np


class PLWE:
    def __init__(self, m, n, q, std) -> None:
        self.m = m
        self.n = n
        self.q = q
        self.std = std
        self.f = Polynomial.basis(n) + 1

    @staticmethod
    def randomPoly(n: int, q: int) -> Polynomial:
        A: np.ndarray = rand(n)*q
        A = A.round(0).astype(int)
        polyA = Polynomial(A)
        return polyA

    def errorPoly(self, n:int) -> Polynomial:
        e: np.ndarray = randn(n)*self.std
        e = e.round(0)
        return Polynomial(e)

    def errorMatrix(self, m:int, n:int):
        eM = np.array([[self.errorPoly(n)] for i in range(m)])
        return eM


    def mod(self, poly:Polynomial) -> Polynomial:
        """
            make polynomial still be ring
        """
        poly = poly % self.f
        poly.coef = poly.coef % self.q
        return poly

    def matrixmod(self, M):
        for i in range(len(M)):
          M[i][0] = self.mod(M[i][0])
        return M

    def polyToMatrix(self, poly:Polynomial)->np.ndarray:
        """
            print the matrix representation of polynomial ring
            ** some problem occur when degree != n
        """
        x = Polynomial([0,1])
        ### some proble occur when degree != n
        A = np.array([poly])
        for i in range(self.m-1):
            poly = poly*x
            poly = self.mod(poly)
            A = np.vstack([A, poly])
        return A

    def keyGen(self):
        #construct a matrix with m polynomials as public key(pk1) -> (m x 1)
        # A = self.randomPoly(self.n, self.q)
        # Amatrix = self.polyToMatrix(A)
        Amatrix = []
        Amatrix = np.array([[self.randomPoly(self.n, self.q)] for i in range(self.m)])
        
        #secret key is a polynomial -> (1 x 1)
        self._S = np.array([[self.errorPoly(self.n)]])
        # print("sk:", self._S)

        #error matrix -> (m x 1)
        ematrix = self.errorMatrix(self.m, self.n)
        # print("e: ",ematrix)

        #public key2(pk2) is pk1*sk+e -> (m x 1)
        T = self.matrixmod(np.dot(Amatrix,self._S) + ematrix ) 
        
        
        # T =  [a + b for a, b in zip(T, ematrix)]
        # T = self.matrixmod(T)
        # print("T:", T)
        return (Amatrix, T)

    def enc(self, pk, m:list):
        M = np.array([[Polynomial(m) * (self.q//2)]])
        A, T = pk
        rmatrix = np.transpose(self.errorMatrix(self.m,self.n))
        e1 = self.errorMatrix(1, self.n)
        e2 = self.errorMatrix(1, self.n)

        U = np.dot(rmatrix,A) + e1
        U = self.matrixmod(U)

        C = self.matrixmod(np.dot(rmatrix,T) + e2 + M)

        return (U, C)

    def dec(self, cipher):
        U, C = cipher
        M = C - np.dot(U, self._S)
        M = self.matrixmod(M)
        # print("M: ",M)
        m = np.where(((M[0][0].coef>self.q//4)&(M[0][0].coef<self.q*3//4)),1,0)
        # m = np.round(M.coef/(self.q/2), 0)
        if len(m) < self.n:
          m = np.append(m, [0 for i in range(self.n - len(m))])
        return m.astype(int)

if __name__ == "__main__":
    m=4
    n=20
    q=3329
    std = 1
    # seed(7)
    alice_bits = randint(0,2,n)
    PLWE_test = PLWE(m,n,q,std)
    pk = PLWE_test.keyGen()
    cipher = PLWE_test.enc(pk, alice_bits)
    U, C = cipher

    bob_bits = PLWE_test.dec(cipher)
    eve_bits = np.where(((C[0][0].coef>q//4)&(C[0][0].coef<q*3//4)),1,0)
    print("---result---")
    print("alice_bits:", alice_bits)
    print("bob_bits:", bob_bits)
    print("eve_bits:", eve_bits)

    # if all(e1 == e2 for e1,e2 in zip(alice_bits,bob_bits)):
    #     print("alice_bits = bob_bits")
    # else:
    #     print("alice_bits != bob_bits")

    errorrate = np.mean( alice_bits != bob_bits )
    print("BER = ", errorrate)

        # m = np.round(M.coef/(self.q/2), 0)
    if len(eve_bits) < n:
        eve_bits = np.append(eve_bits, [0 for i in range(n - len(eve_bits))])

    # if all(e1 == e2 for e1,e2 in zip(alice_bits,eve_bits)):
    #     print("alice_bits = eve_bits")
    # else :
    #     print("alice_bits != eve_bits")

    errorrate = np.mean( alice_bits != eve_bits )
    print("BER = ", errorrate)