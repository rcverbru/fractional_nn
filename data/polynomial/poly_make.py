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

# prepare data
# x = matrix of polynomials
# y = matrix of corresponding derivatives
# half = matrix of half derivative
# coefs = polynomial coefficients, orders = poly orders
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

# make the rest of the training polynomials
for count in range(N):
    coefs = (np.random.rand(terms)-0.5)*4
    orders = np.random.randint(0,5,terms)
    print("Count: ", count, "Polynomial Orders: ", orders)
    newx = np.zeros((1,D_in))
    newy = np.zeros((1,D_in))
    newhalf = np.zeros((1,D_in))
    for term in range(terms):
        newx += coefs[term]*t**orders[term]
        # newhalf += coefs[term]*gamma(orders[term]+1)\
                # /gamma(orders[term]+1-0.5)*t**(orders[term]-0.5)
        if orders[term] > 0:
            newhalf += coefs[term]*gamma(orders[term]+1)\
                    /gamma(orders[term]+1-0.5)*t**(orders[term]-0.5)
            newy += coefs[term]*orders[term]*t**(orders[term]-1)
    x = np.append(x,newx,axis=0)
    y = np.append(y,newy,axis=0)
    half = np.append(half,newhalf,axis=0)

# save the input and output data (numpy.save)
np.save('x.npy', x)
np.save('y.npy', y)
np.save('half.npy', half)