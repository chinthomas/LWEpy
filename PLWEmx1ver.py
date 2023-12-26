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
        print("A: ",A[0])
        return A

    def keyGen(self):
        #construct a matrix with m polynomials as public key(pk1) -> (m x 1)
        A = self.randomPoly(self.n, self.q)
        Amatrix = self.polyToMatrix(A)

        #secret key is a polynomial -> (1 x 1)
        self._S = self.errorPoly(self.n)
        print("sk:", self._S)

        #error matrix -> (m x 1)
        ematrix = self.errorMatrix(self.m, self.n)
        print("e: ",ematrix)

        #public key2(pk2) is pk1*sk+e -> (m x 1)
        T = self.matrixmod(np.dot(Amatrix,self._S))
        T =  [a + b for a, b in zip(T, ematrix)]
        T = self.matrixmod(T)
        return (A, T)

    def enc(self, pk, m:list):
        M = Polynomial(m) * (self.q//2)
        A, T = pk
        r = self.errorPoly(self.n)
        rmatrix = np.transpose(self.polyToMatrix(r))
        e1 = self.errorPoly(self.n)
        e2 = self.errorPoly(self.n)

        U = np.dot(rmatrix,A)[0][0] + e1
        U = self.mod(U)

        C = self.mod(np.dot(rmatrix,T)[0][0] + e2 + M)

        return (U, C)

    def dec(self, cipher):
        """
            ** how to rounding decoding message ???
        """
        U, C = cipher
        print("U: ",U.coef)
        print("C: ",C.coef)
        M = C - U*self._S
        M = self.mod(M)
        print("M: ",M.coef)
        m = np.where(((M.coef>self.q//4)&(M.coef<self.q*3//4)),1,0)
        # m = np.round(M.coef/(self.q/2), 0)
        return m.astype(int)

if __name__ == "__main__":
    m=4
    n=16
    q=37
    std = 1
    seed(18)
    m1 = randint(0,2,n)
    PLWE_test = PLWE(m,n,q,std)
    pk = PLWE_test.keyGen()
    cipher = PLWE_test.enc(pk, m1)
    U, C = cipher
    m2 = PLWE_test.dec(cipher)
    print("m1:", m1)
    print("m2:", m2)
    if all(e1 == e2 for e1,e2 in zip(m1,m2)):
      print("m1 = m2")

    errorrate = np.mean( m1 != m2 )
    print("BER = ", errorrate)
