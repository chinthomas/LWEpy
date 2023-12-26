from Crypto.Cipher import DES
from PLWE import PLWE
from numpy.polynomial import Polynomial
from numpy.random import*
import numpy as np


def pad(text):
  n = len(text) % 8
  return text + (b' ' * n)

def createKey(m):
    """
        m need to larger than 64
    """
    key = [sum(m[8*i:8*(i+1)]) for i in range(8)]
    key = ''.join(map(str,key)).encode('utf-8')
    return key

if __name__ == "__main__":
    m=4
    n=64
    q=1289
    std = 2
    seed(318)

    ml = 1024
    
    # encode
    key = randint(0,2,n).astype(int)
    PLWE_test = PLWE(m,n,q,std)
    pk = PLWE_test.keyGen()
    cipher = PLWE_test.enc(pk, key)
    U, C = cipher
    key = createKey(key)
    print("key in bytes:", key)
    message = randint(0,2,ml)
    message = ''.join(map(str,message)).encode('utf-8')
    des = DES.new(key, DES.MODE_ECB)
    cipher_text = des.encrypt(message)

    # decode

    key2 = PLWE_test.dec(cipher)
    key2 = createKey(key2)
    print("key2 in bytes:", key)

    des2 = DES.new(key2, DES.MODE_ECB)
    cipher_text2 = cipher_text.hex()
    message2 = des2.decrypt(cipher_text)

    print("message to send: ", message)
    print("message receieved: ", message2)
    errorrate = np.mean( message != message2 )
    print("BER = ", errorrate)

    m_c_rate = len(cipher_text2)/len(message)
    print("message＿cypher rate:　",m_c_rate)
