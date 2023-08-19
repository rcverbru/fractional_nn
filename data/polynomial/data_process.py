# Data
import numpy as np
import matplotlib.pyplot as plt

# arr_prf = np.load('arr_prf_1.npy')
# print(arr_prf.shape)
# arr_prf_app = np.load('arr_prf_2.npy')
# print(arr_prf_app.shape)
# arr_prf = np.append(arr_prf[1:, :], arr_prf_app[1:, :], axis = 0)
# print(arr_prf.shape)
# row = len(arr_prf)

arr_prf = np.load('arr_prf.npy')
arr_prf = arr_prf[1:, :]
row = len(arr_prf)

x_range = np.arange(0, row)
print(x_range.shape)

ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)

plt.subplot(ax1)
plt.plot(x_range, arr_prf[0:, 0:2])
plt.ylabel('Errors')
plt.legend(['train', 'test'])
plt.subplot(ax2)
plt.plot(x_range, arr_prf[0:, 2:])
plt.legend(['train - output', 'test - output'])
plt.savefig('errors_modelt.png')
plt.show() # after plt.show() is called, a new figure is created

# # zoom in some part of the figure
# x_range = np.arange(20, 265)
# print(x_range.shape)

# ax1 = plt.subplot(1, 2, 1)
# ax2 = plt.subplot(1, 2, 2)

# plt.subplot(ax1)
# plt.plot(x_range, arr_prf[20:265, 0:2])
# plt.ylabel('Errors')
# plt.legend(['train', 'test'])
# plt.subplot(ax2)
# plt.plot(x_range, arr_prf[20:265, 2:])
# plt.legend(['train - output', 'test - output'])

# plt.show()
# plt.savefig('zoomin', format = 'png')
