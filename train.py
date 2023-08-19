import torch
import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt
from tempfile import TemporaryFile
import plotly.express as px

weight_file = input("Weight file to use: ")
path = "/home/rcv/dev/fractional/" + weight_file
print(path)

class SymmetricNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(SymmetricNet, self).__init__()
        self.hidden = torch.nn.Linear(D_in, H)
        self.output_linear = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.hidden(x)
        h_relu = h_relu.clamp(min=0)
        h_relu = self.output_linear(h_relu)
        h_middle = h_relu

        h_relu = self.hidden(x)
        h_relu = h_relu.clamp(min=0)
        y_pred = self.output_linear(h_relu)
        
        return y_pred, h_middle

# turn on interactive plotting
plt.ion()

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H  = 10000, 200, 400
D_out = D_in
# terms is the number of terms in the polynomial
terms = 6

# Set number of epochs
num_epoch = int(input("Number of Epochs: "))
if num_epoch is None:
    num_epoch = 20

# t = time vector
t = np.linspace(0.01,2,D_in)

# loading data
x = np.load('x.npy')
y = np.load('y.npy')
half = np.load('half.npy')

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

# TRAINING

# Construct our model by instantiating the class defined above
model = SymmetricNet(D_in, H, D_out)

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
            model_outs = model(x_train)
            y_pred = model_outs[0]
            
            # loss 2
            h_middle_shift = model_outs[1][0:, 1:]

            # compute the error
            loss1 = criterion(y_pred, y_train)
            
            loss2 = criterion(model_outs[1][0:, 0:-1], h_middle_shift)
            
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
            train_num = np.arange(tt-9, dtt+1, 1)
            arr_prf_train = arr_prf[-10:, 0]
            arr_prf_test = arr_prf[-10:, 1]
            # trigger to avoid overfitting or the cases when the training fails to converge
            if all(i < j for i, j in zip(arr_prf_test, arr_prf_test[1:])):
                break
            plt.subplot(ax1) # make ax1 "current"
            plt.plot(train_num, arr_prf_train, label = 'training')
            plt.plot(train_num, arr_prf_test, label = 'test')
            fig = px.line(, x="train_num", y="arr_prf_train", title="Training")
            fig.show()
            plt.title(tt)
            plt.legend()
            plt.grid()
        

        # pick an input for test
        choose = np.random.randint(0, 9*N//10-1)
        myx = x[choose].numpy()  

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
        # do second network
        h_relu = model.hidden(h_relu)
        h_relu = h_relu.clamp(min=0)
        myy = model.output_linear(h_relu)
        # plot output layer value
        plt.plot(t,myy.detach().numpy(), label='deriv')
        # plot exact derivative
        plt.plot(t,y[choose].numpy(), label='exact deriv')    

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
    torch.save(model.state_dict(), path)