from Crypto.Cipher import DES
from PLWE import PLWE
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
    # seed(318)
    m=4     # number of polynomial in matrix
    n=64    # share key length
    q=1289
    std = 2
    ml = 1024   # message length

    # 1. exchange the share key via public key encryption system
    # ----- Alice -----
    # Alice generate a share key
    alice_key = randint(0,2,n).astype(int)

    # use private key system to transform the share key
    PLWE_test = PLWE(m,n,q,std)
    pk = PLWE_test.keyGen()
    cipher = PLWE_test.enc(pk, alice_key)

    # DES package need 8 bytes key
    alice_share_key = createKey(alice_key)
    print("alice's share key in bytes:", alice_share_key)

    # ----- Bob -----
    bob_key = PLWE_test.dec(cipher)
    bob_share_key = createKey(bob_key)
    print("Bob's share key in bytes:", bob_share_key)
    

    # 2. with share key they can transform message 
    # ----- Alice -----
    des = DES.new(alice_share_key, DES.MODE_ECB)

    # encode message
    message = randint(0,2,ml)
    # message = input("the message to transform: ")
    message = ''.join(map(str,message)).encode('utf-8')
    cipher_text = des.encrypt(message)
    
    # ----- Bob -----
    des2 = DES.new(bob_share_key, DES.MODE_ECB)

    # decode message
    cipher_text2 = cipher_text.hex()
    message2 = des2.decrypt(cipher_text)


    # 3. result
    print("-----result-----")
    print("message to send: ", message.decode('utf-8'))
    print("message receieved: ", message2.decode('utf-8'))
    errorrate = np.mean( message != message2 )
    print("BER = ", errorrate)

    m_c_rate = len(cipher_text2)/len(message)
    print("message-cypher rate: ",m_c_rate)
