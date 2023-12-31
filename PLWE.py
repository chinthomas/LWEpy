from numpy.polynomial import Polynomial
from numpy.random import*
import numpy as np
class PLWE:
    def __init__(self, n, q) -> None:
        self.n = n
        self.q = q
        self._S = Polynomial(np.zeros(n))
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
        print(poly.coef)
        for i in range(self.n):
            poly = poly*x
            poly = self.mod(poly)
            print(poly.coef)
            # A = np.vstack([A, poly.coef.astype(int)])
        # print(A)

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
        M = Polynomial(m) * np.round(self.q/2)
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
            if i < np.round(self.q*3/4) and i > np.round(self.q/4):
                m.append(1)
            else:
                m.append(0)
        return np.array(m).astype(int)

if __name__ == "__main__":
    n=7
    q=17
    msg_bits = randint(0,2,n)
    seed(0)
    PLWE_test = PLWE(n,q)
    A, T = PLWE_test.keyGen()
    # S = PLWE_test.getSK()
    cipher = PLWE_test.enc((A,T), msg_bits)
    U, C = cipher
    dec_bits = PLWE_test.dec(cipher)
    # eve
    eve_machine = PLWE(n,q)
    eve_bits = eve_machine.dec(cipher)
    # print("--cipher--")
    # print("C", C)
    # print("U", U)
    print("polynomial ring represents a matrix")
    PLWE_test.polyToMatrix(A)
    print("")
    print("---result---")
    print("msg_bits:", msg_bits)
    print("eve_bits:", np.array(eve_bits))
    print("dec_bits:", dec_bits)
    # PLWE_test.polyToMatrix(T)
    print("Decode BER",np.mean(msg_bits != dec_bits))
    print("eve BER", np.mean(msg_bits != eve_bits))