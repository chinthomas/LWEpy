from numpy.polynomial import Polynomial
from numpy.random import*
import numpy as np
class PLWE:
    def __init__(self, n, q) -> None:
        self.n = n
        self.q = q
        self.f = Polynomial.basis(n) + 1
    
    @staticmethod
    def randomPoly(n: int, q: int) -> Polynomial:
        A: np.ndarray = rand(n)*q
        A = A.round(0).astype(int)
        polyA = Polynomial(A)
        return polyA
    @staticmethod
    def errorPoly(n:int) -> Polynomial:
        e: np.ndarray = rand(n).round(0).astype(int)
        return Polynomial(e)
    
    def mod(self, poly:Polynomial) -> Polynomial:
        """
            make polynomial still be ring
        """
        poly = poly % self.f
        poly.coef = poly.coef % self.q
        return poly
    
    def polyToMatrix(self, poly:Polynomial)->np.ndarray:
        """
            print the matrix representation of polynomial ring
            ** some proble occur when degree != n
        """
        x = Polynomial([0,1])
        ### some proble occur when degree != n
        A = poly.coef.astype(int)
        print(poly)
        for i in range(self.n):
            poly = poly*x
            poly = self.mod(poly)
            A = np.vstack([A, poly.coef.astype(int)])
        print(A)


    def keyGen(self):
        A = self.randomPoly(self.n, self.q)
        self.S = self.randomPoly(self.n, self.q)
        e = self.errorPoly(self.n)
        T = A * self.S + e
        T = self.mod(T)
        return A, T

    def enc(self, A:Polynomial, T:Polynomial, m):
        r = self.randomPoly(self.n, self.q)
        e1 = self.errorPoly(self.n)
        e2 = self.errorPoly(self.n)
        U = A * r + e1
        U = self.mod(U)
        C = T * r + e2 + self.q//2*m
        C = self.mod(C)
        return U, C

    def dec(self, U:Polynomial, C:Polynomial):
        """
            ** how to rounding decoding message ???
        """
        M = C - U * self.S
        M = self.mod(M)
        print(M)
        return M

if __name__ == "__main__":
    n=7
    q=13
    seed(0)
    PLWE_test = PLWE(n,q)
    A, T = PLWE_test.keyGen()
    U, C = PLWE_test.enc(A, T, 1)
    M = PLWE_test.dec(U,C)
    # PLWE_test.polyToMatrix(A)
    # PLWE_test.polyToMatrix(T)