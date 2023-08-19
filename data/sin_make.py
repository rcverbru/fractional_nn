import numpy as np
from scipy.special import gamma

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H  = 10000, 200, 400
D_out = D_in
# terms is the number of terms in the polynomial
terms = 6
# t = time vector
t = np.linspace(0.01,2,D_in)

coefs = (np.random.rand(terms)-0.5)*4 # creates random values between [-2 2]
orders = np.random.randint(0,5,terms) # pick from 0, 1, 2, 3, 4 randomly
x = np.zeros((1,D_in))
y = np.zeros((1,D_in))
half = np.zeros((1,D_in))

# make the first polynomial
for term in range(terms):
    x += coefs[term]*t**orders[term]
    # half derivative with RL definition
    # half += coefs[term]*gamma(orders[term]+1)/gamma(orders[term]+1-0.5)\
            # *t**(orders[term]-0.5)
    if orders[term] > 0:
        # half derivative with Caputo definition
        half += coefs[term]*gamma(orders[term]+1)/gamma(orders[term]+1-0.5)\
                *t**(orders[term]-0.5)
        y += coefs[term]*orders[term]*t**(orders[term]-1)

for count in range(N):
    coef = np.random.randint(0,1)
    a = np.random.randint(-2,2)
    b = np.random.randint(0,10)
    c = np.random.randint(-5,5)
    d = np.random.randint(-5,5)
    print("Sin/Cos (0/1): ", coef, "Yes: ", a, b, c, d)
    


np.save('x.npy', x)
np.save('y.npy', y)
np.save('half.npy', half)