# Train on the full derivative to obtain the half derivative
import torch
import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt
from tempfile import TemporaryFile

####### replace the path ############
path = "/home/rcv/dev/fractional/weights.pt" 

class SymmetricNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(SymmetricNet, self).__init__()
        self.hidden = torch.nn.Linear(D_in, H)
        self.output_linear = torch.nn.Linear(H, D_out)

    def forward(self, x):
        # compute the ouptput of the input layer (or input to the first hidden layer)
        h_relu = self.hidden(x)
        # compute output of hidden layer. clamp is the ReLU function
        h_relu = h_relu.clamp(min=0)
        # compute the output
        h_relu = self.output_linear(h_relu)
        # extract the middle layer for later use
        h_middle = h_relu
        # this does the same thing with the same layers
        h_relu = self.hidden(h_relu)
        h_relu = h_relu.clamp(min=0)
        y_pred = self.output_linear(h_relu)
        return y_pred, h_middle


# # convert a function defined by zeros to a function with coefficients
# def zeros_to_coefs(all_zeros):
#     return coefs, orders 

# turn on interactive plotting
plt.ion()

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H  = 10000, 200, 400
D_out = D_in
# terms is the number of terms in the polynomial
terms = 6

# Set number of epochs
num_epoch = 10

# t = time vector
t = np.linspace(0.01,2,D_in)

##################################################################################
# # prepare data
# # x = matrix of polynomials
# # y = matrix of corresponding derivatives
# # half = matrix of half derivative
# # coefs = polynomial coefficients, orders = poly orders
# coefs = (np.random.rand(terms)-0.5)*4 # creates random values between [-2 2]
# orders = np.random.randint(0,5,terms) # pick from 0, 1, 2, 3, 4 randomly
# x = np.zeros((1,D_in))
# y = np.zeros((1,D_in))
# half = np.zeros((1,D_in))
# # make the first polynomial
# for term in range(terms):
#     x += coefs[term]*t**orders[term]
#     # half derivative with RL definition
#     # half += coefs[term]*gamma(orders[term]+1)/gamma(orders[term]+1-0.5)\
#             # *t**(orders[term]-0.5)
#     if orders[term] > 0:
#         # half derivative with Caputo definition
#         half += coefs[term]*gamma(orders[term]+1)/gamma(orders[term]+1-0.5)\
#                 *t**(orders[term]-0.5)
#         y += coefs[term]*orders[term]*t**(orders[term]-1)

# # make the rest of the training polynomials
# for count in range(N):
#     coefs = (np.random.rand(terms)-0.5)*4
#     orders = np.random.randint(0,5,terms)
#     print(count, orders)
#     newx = np.zeros((1,D_in))
#     newy = np.zeros((1,D_in))
#     newhalf = np.zeros((1,D_in))
#     for term in range(terms):
#         newx += coefs[term]*t**orders[term]
#         # newhalf += coefs[term]*gamma(orders[term]+1)\
#                 # /gamma(orders[term]+1-0.5)*t**(orders[term]-0.5)
#         if orders[term] > 0:
#             newhalf += coefs[term]*gamma(orders[term]+1)\
#                     /gamma(orders[term]+1-0.5)*t**(orders[term]-0.5)
#             newy += coefs[term]*orders[term]*t**(orders[term]-1)
#     x = np.append(x,newx,axis=0)
#     y = np.append(y,newy,axis=0)
#     half = np.append(half,newhalf,axis=0)

# # save the input and output data (numpy.save)
# np.save('x.npy', x)
# np.save('y.npy', y)
# np.save('half.npy', half)

#########################################################################
# # prepare data - generate random functions by using zeros
# # x = matrix of polynomials
# # y = matrix of corresponding derivatives
# # half = matrix of half derivative
# x = np.zeros((1, D_in))
# y = np.zeros((1, D_in))
# half = np.zeros((1, D_in))
# # make the first polynomial
# order_max = np.random.randint(1, 5) # generate a random int from [1 5)
# print(order_max)
# roots = (np.random.rand(order_max) + 0.01/1.99) * 1.99 # zeros between [0.01 2)
# print(roots)
# # convert the function defined by zeros to a function with coefficients
# coefs = np.poly(roots)
# orders = np.arange(len(coefs)-1, -1, -1)
# for term in range(len(coefs)):
#     x += coefs[term]*t**orders[term]
#     # half derivative with RL definition
#     # half += coefs[term]*gamma(orders[term]+1)/gamma(orders[term]+1-0.5)\
#             # *t**(orders[term]-0.5)
#     if orders[term] > 0:
#         # half derivative with Caputo definition
#         half += coefs[term]*gamma(orders[term]+1)/gamma(orders[term]+1-0.5)\
#                 *t**(orders[term]-0.5)
#         y += coefs[term]*orders[term]*t**(orders[term]-1)

# # make the rest of the training polynomials
# for count in range(N):
#     order_max = np.random.randint(1, 5)
#     roots = (np.random.rand(order_max) + 0.01/1.99) * 1.99 
#     coefs = np.poly(roots)
#     orders = np.arange(len(coefs)-1, -1, -1) 
#     print(count, len(coefs), orders)
#     newx = np.zeros((1,D_in))
#     newy = np.zeros((1,D_in))
#     newhalf = np.zeros((1,D_in))
#     for term in range(len(coefs)):
#         newx += coefs[term]*t**orders[term]
#         # newhalf += coefs[term]*gamma(orders[term]+1)\
#                 # /gamma(orders[term]+1-0.5)*t**(orders[term]-0.5)
#         if orders[term] > 0:
#             newhalf += coefs[term]*gamma(orders[term]+1)\
#                     /gamma(orders[term]+1-0.5)*t**(orders[term]-0.5)
#             newy += coefs[term]*orders[term]*t**(orders[term]-1)
#     x = np.append(x,newx,axis=0)
#     y = np.append(y,newy,axis=0)
#     half = np.append(half,newhalf,axis=0)

# # save the input and output data (numpy.save)
# np.save('x.npy', x)
# np.save('y.npy', y)
# np.save('half.npy', half)

#########################################################################
# if you have the data, just load it
# loading data
x = np.load('x.npy')
y = np.load('y.npy')
half = np.load('half.npy')

# shuffle data
# z = np.append(x, y, axis = 1)
# np.random.shuffle(z)
# x = z[:, 0:D_in]
# y = z[:, D_in:]
# split data into training (0.9) and test (0.1) sets 
x_train = x[0:9*N//10, :]
y_train = y[0:9*N//10, :]
half_train = half[0:9*N//10, :]
x_test = x[9*N//10:, :]
y_test = y[9*N//10:, :]
half_test = half[9*N//10:, :]

# convert matrices to the right data type for nnet input and output
x = torch.FloatTensor(x)
y = torch.FloatTensor(y)
x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train)
half_train = torch.FloatTensor(half_train)
x_test = torch.FloatTensor(x_test)
half_test = torch.FloatTensor(half_test)

#################################################################################
# training
# Construct our model by instantiating the class defined above
model = SymmetricNet(D_in, H, D_out)
# # if you have the model, just load it
# model.load_state_dict(torch.load(path))
# model.eval()
# print("load model:")
# print(model)

# optimize using mean square error
criterion = torch.nn.MSELoss()

# use the stochastic gradient descent method
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.2)

# initialization
loss = 1e7
half_err_train = 1e7
arr_prf = np.zeros((1, 2)) # any good method to do this?

def train():
    global arr_prf
    # outer loop, just to frequently plot
    for tt in range(num_epoch):  # trigger
    # do 100 iterations between plots
        for stepno in range(100):
            # Forward pass: Compute predicted y by passing x to the model
            # model(x) computes the output based on input x
            model_outs = model(x_train)
            y_pred = model_outs[0]
            
            # loss 2
            h_middle_shift = model_outs[1][0:, 1:]

            # loss 3
            # h_middle_diff = model_outs[1][0:, 1:] - model_outs[1][0:, 0:-1]
            # h_middle_var = torch.mul(h_middle_diff[0:, 1:], h_middle_diff[0:, 0:-1])
            # h_middle_var[h_middle_var >= 0] = 0
            # h_middle_var[h_middle_var < 0] = 1
            # mat_ref = torch.zeros(h_middle_var.shape)    

            # loss 4
            # dat_middle = model_outs[1].detach().numpy()
            # dat_xform = dat_middle/np.sqrt(t)
            # coef_fit = np.polynomial.polynomial.polyfit(t, dat_xform.T, 3)
            # val_fit = np.polynomial.polynomial.polyval(t, coef_fit)
            # loss4 = np.sum((val_fit - dat_xform)**2)
            # print(loss4)    
            
            # compute the error
            loss1 = criterion(y_pred, y_train)
            
            loss2 = criterion(model_outs[1][0:, 0:-1], h_middle_shift)
            
            # loss3 = criterion(h_middle_var, mat_ref)
            # print(loss3.item())
            loss = loss1 + loss2
            print("Epoch ", tt, '\n',
                  "Loss 1: ", loss1.item(), '\n',
                  "Loss 2: ", loss2.item(), '\n',
                  "Run: ", stepno, "Total Loss: ", loss.item(), '\n')

            # define prf metric by the error in predicting the half derivative
            half_err_train = criterion(model_outs[1], half_train)

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()  # clears old gradients from the last step (otherwise you’d just accumulate the gradients from all loss.backward() calls)
            loss.backward()  # computes the derivative of the loss w.r.t. the parameters (or anything requiring gradients) using backpropagation
            # print(model.hidden.weight.grad)
            optimizer.step()  # causes the optimizer to take a step based on the gradients of the parameters

        plt.close()    

        # define prf metric
        print('Training error:')
        print(half_err_train.detach().numpy())
        model_outs_test = model(x_test)
        half_err_test = criterion(model_outs_test[1], half_test)
        print('Test error:')
        print(half_err_test.detach().numpy())

         # save the prf metrics after every 200 epochs
        arr_prf = np.append(arr_prf, [[half_err_train.detach().numpy(), half_err_test.detach().numpy()]], axis = 0)
        print(arr_prf)

        ax1 = plt.subplot(1, 2, 1)
        ax2 = plt.subplot(1, 2, 2)

        if tt >= 10:
            train_num = np.arange(tt-9, tt+1, 1)
            arr_prf_train = arr_prf[-10:, 0]
            arr_prf_test = arr_prf[-10:, 1]
            # trigger to avoid overfitting or the cases when the training fails to converge
            if all(i < j for i, j in zip(arr_prf_test, arr_prf_test[1:])):
                break
            plt.subplot(ax1) # make ax1 "current"
            plt.plot(train_num, arr_prf_train, label = 'training')
            plt.plot(train_num, arr_prf_test, label = 'test')
            plt.title(tt)
            plt.legend()
            plt.grid()
        

        # pick an input for test
        choose = np.random.randint(0, 9*N//10-1)
        myx = x[choose].numpy()  
        
        # # myx = 0.72*t**4 + 0.5*t**2 - t
        # # myy_exact = 0.72*4*t**3 + 0.5*2*t - 1
        # # myx_half_RL = 0.72*gamma(4+1)/gamma(4+1-0.5)*t**(4-0.5)+0.5*gamma(2+1)/gamma(2+1-0.5)*t**(2-0.5)\
        # #             -gamma(1+1)/gamma(1+1-0.5)*t**(1-0.5)+5*gamma(0+1)/gamma(0+1-0.5)*t**(0-0.5)
        # # myx_half_caputo = 0.72*gamma(4+1)/gamma(4+1-0.5)*t**(4-0.5)+0.5*gamma(2+1)/gamma(2+1-0.5)*t**(2-0.5)\
        # #             -gamma(1+1)/gamma(1+1-0.5)*t**(1-0.5)
        
        plt.subplot(ax2)
        plt.plot(t,myx, label='input')
        # go through first three steps
        h_relu = model.hidden(torch.FloatTensor(myx))
        h_relu = h_relu.clamp(min=0)
        h_relu = model.output_linear(h_relu)
        # plot middle layer value
        plt.plot(t,h_relu.detach().numpy(),label='half') 
        # plot exact half derivative
        plt.plot(t,half[choose],label='half exact')
        # plt.plot(t,myx_half_caputo,label='half exact')
        # do second network
        h_relu = model.hidden(h_relu)
        h_relu = h_relu.clamp(min=0)
        myy = model.output_linear(h_relu)
        # plot output layer value
        plt.plot(t,myy.detach().numpy(), label='deriv')
        # plot exact derivative
        plt.plot(t,y[choose].numpy(), label='exact deriv')
        # plt.plot(t,myy_exact, label='exact deriv')      

        plt.title(tt)
        plt.legend()
        plt.grid()
        plt.pause(1)
        plt.savefig(str(tt), format = 'png')
        
        if loss.item() < 1e-2: # trigger 
            break

try:
    train()
finally:
# except KeyboardInterrupt:
    torch.save(model.state_dict(), path)
