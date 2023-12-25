from numpy.polynomial import Polynomial
from numpy.random import*
import numpy as np
class RLWE:
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
        e: np.ndarray = rand(n)
        e = e.round(0).astype(int)
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

    def getSK(self):
        return self._S
    
    def keyGen(self):
        # discrete random ring polynominal
        A = self.randomPoly(self.n, self.q)
        # error distribution
        self._S = self.errorPoly(self.n)
        # error distribution
        e = self.errorPoly(self.n)

        T = A * self._S
        T = self.mod(T) + e
        return (A, T)

    def enc(self, pk:(Polynomial,Polynomial), m:list):
        M = Polynomial(m) * (self.q//2)
        A, T = pk
        r = self.errorPoly(self.n)
        e1 = self.errorPoly(self.n)
        e2 = self.errorPoly(self.n)

        U = A * r + e1
        U = self.mod(U)

        C = T * r + e2 + M
        C = self.mod(C)
        return (U, C)

    def dec(self, cipher:(Polynomial,Polynomial)) -> np.ndarray:
        """
            ** how to rounding decoding message ???
        """
        U, C = cipher
        M = C - (U * self._S)
        # M.coef = M.coef % self.q
        M = self.mod(M)
        m = []
        for i in M.coef:
            if i < self.q*3//4 and i > self.q//4:
                m.append(1)
            else:
                m.append(0)
        return np.array(m).astype(int)

if __name__ == "__main__":
    n=7
    q=43
    m1 = np.array([1,0,0,1,0,1,0])
    seed(1)
    PLWE_test = RLWE(n,q)
    A, T = PLWE_test.keyGen()
    S = PLWE_test.getSK()
    cipher = PLWE_test.enc((A,T), m1)
    U, C = cipher
    m2 = PLWE_test.dec(cipher)
    print("m1:", m1)
    print("--cipher--")
    print("U", U)
    print("C", C)
    print("M", PLWE_test.mod(C-U*S))
    print("m2:", m2)
    # PLWE_test.polyToMatrix(A)
    # PLWE_test.polyToMatrix(T)