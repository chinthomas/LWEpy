# LWEpy
* GitHub link: https://github.com/chinthomas/LWEpy
* The project is to simulate how LWE, RLWE ... Latticed-Base PQCs work.
* Before using the code, you need to install some packages.
```
pip install -r requirements.txt
```

#### LWE.py
Run LWE.py and need to give the parameters n, q, and message bit length to see the result.
```
> n to be : <enter a number>
> q to be : <enter a number>
> message bits length : <enter a number>
```
#### PLWE.py
Run PLWE.py to see the result.
Can modify the parameters in the script.
```
n=7
q=17
```

#### KEM.py
Run KEM.py to see the result.
Can modify the parameters in the script.
```
m=4     # number of polynomial in matrix
n=64    # share key length
q=1289
std = 2
ml = 1024   # message length
```
