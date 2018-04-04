import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data[:, :2]  # Only the first two features
Y = iris.target

# Make train and test split
np.random.seed(1337)
idx = np.random.permutation(150)
X, Y = X[idx], Y[idx]

X_tr, Y_tr = X[:75], Y[:75]
X_tst, Y_tst = X[75:], Y[75:]

def get_train_data():
    return X_tr, Y_tr

def get_test_data():
    return X_tst, Y_tst


if __name__ == "__main__":
    # Plot training and test data
    plt.scatter(X_tst[:, 0], X_tst[:, 1], c=Y_tst)
    plt.figure()

    plt.scatter(X_tr[:, 0], X_tr[:, 1], c=Y_tr)
    plt.show()

