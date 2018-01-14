import numpy as np
from mnist import MNIST
mndata = MNIST("C:/Users/Yuki/Program/projects/le4/Class-le4-image/MNIST/")
X, Y = mndata.load_training()
X = np.array(X)
X = X.reshape((X.shape[0],28,28))
Y = np.array(Y)

import matplotlib.pyplot as plt
from pylab import cm
idx = 100
plt.imshow(X[idx], cmap=cm.gray)
print(Y[idx])
plt.show()
