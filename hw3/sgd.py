#################################
# Your name:
#################################


import numpy as np
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing
import matplotlib.pyplot as plt
import seaborn as sns

"""
Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""


def helper():
    mnist = fetch_openml('mnist_784', as_frame=False)
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

    test_data_unscaled = data[60000 + test_idx, :].astype(float)
    test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def SGD_hinge(data, labels, C, eta_0, T):
    """
    Implements SGD for hinge loss.
    """
    dim = data.shape[1]
    w = np.zeros(dim, dtype=numpy.float64)
    for t in range(1, T + 1):
        eta = eta_0 / t
        i = np.random.randint(0, len(data))  # select a random point
        if labels[i] * np.dot(w, data[i]) < 1:
            w = (1 - eta) * w + eta * C * labels[i] * data[i]
        else:
            w = (1 - eta) * w
    return w


def SGD_log(data, labels, eta_0, T):
    """
    Implements SGD for log loss.
    """
    dim = data.shape[1]
    w = np.zeros(dim, dtype=numpy.float64)
    norms = []
    for t in range(1, T + 1):
        eta = eta_0 / t
        i = np.random.randint(0, len(data))  # select a random point

        x_i = data[i]
        y_i = labels[i]

        exp_val = np.exp(np.dot(y_i * w, x_i))
        w = w + (eta * y_i * x_i) / (exp_val + 1)
        norms.append(np.linalg.norm(w))
    return w, norms


#################################

# Place for additional code

#################################


def test_accuracy(validation_data, validation_labels, w):
    """
    Check the performance of w as a linear separator on the validation set.
    """
    n_success = 0
    for i in range(len(validation_data)):
        dotproduct = np.dot(w, validation_data[i])
        if not np.isnan(dotproduct) and int(np.sign(dotproduct)) == validation_labels[i]:
            n_success += 1
    return n_success / len(validation_data)


def main_hinge(train_data, train_labels, validation_data, validation_labels, test_data, test_labels):
    print("Hinge...")

    # Find best eta_0
    eta_0_list = np.logspace(-1, 0.5, 50)
    eta_0_results = []
    for eta_0 in eta_0_list:
        avg_accuracy = np.average([test_accuracy(validation_data,
                                                 validation_labels,
                                                 SGD_hinge(train_data, train_labels, C=1, eta_0=eta_0, T=1000))
                                   for _ in range(10)])
        eta_0_results.append(avg_accuracy)

    sns.set()
    plt.title("Accuracy by eta_0")
    plt.xlabel("eta_0")
    plt.ylabel("Accuracy")
    plt.ylim(0.8, 1)
    plt.xscale('log')

    plt.plot(eta_0_list, eta_0_results, 'ro')
    plt.show()

    best_eta_0 = eta_0_list[np.argmax(eta_0_results)]
    print(f"Best eta_0: {best_eta_0}")

    # Find best C
    C_list = np.logspace(-5, 5, 50)
    C_results = []
    for C in C_list:
        avg_accuracy = np.average([test_accuracy(validation_data,
                                                 validation_labels,
                                                 SGD_hinge(train_data, train_labels, C=C, eta_0=best_eta_0, T=1000))
                                   for _ in range(10)])
        C_results.append(avg_accuracy)

    sns.set()
    plt.title("Accuracy by C")
    plt.xlabel("C")
    plt.ylabel("Accuracy")
    plt.ylim(0.9, 1)
    plt.xscale('log')

    plt.plot(C_list, C_results, 'ro')
    plt.show()

    best_C = C_list[np.argmax(C_results)]
    print(f"Best C: {best_C}")

    # Final training with best parameters
    w = SGD_hinge(train_data, train_labels, C=best_C, eta_0=best_eta_0, T=20000)
    plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest')
    print(f"Accuracy on test: {test_accuracy(test_data, test_labels, w)}")


def main_log(train_data, train_labels, validation_data, validation_labels, test_data, test_labels):
    print("Log...")

    # Find best eta_0
    eta_0_list = np.logspace(-5, 5, 20)
    eta_0_results = []
    for eta_0 in eta_0_list:
        avg_accuracy = np.average([test_accuracy(validation_data,
                                                 validation_labels,
                                                 SGD_log(train_data, train_labels, eta_0=eta_0, T=1000)[0])
                                   for _ in range(10)])
        eta_0_results.append(avg_accuracy)

    sns.set()
    plt.title("Accuracy by eta_0")
    plt.xlabel("eta_0")
    plt.ylabel("Accuracy")
    plt.ylim(0.8, 1)
    plt.xscale('log')

    plt.plot(eta_0_list, eta_0_results, 'ro')
    plt.show()

    best_eta_0 = eta_0_list[np.argmax(eta_0_results)]
    print(f"Best eta_0: {best_eta_0}")

    # Final training with best parameters
    w, norms = SGD_log(train_data, train_labels, eta_0=best_eta_0, T=20000)
    plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest')
    print(f"Accuracy on test: {test_accuracy(test_data, test_labels, w)}")

    sns.set()
    plt.title("Norm as a function of iteration number")
    plt.xlabel("t (iteration number)")
    plt.ylabel("||w_t||")

    plt.plot(list(range(1, 20000 + 1))[2:], norms[2:])
    plt.show()


def main():
    data = helper()
    main_hinge(*data)
    main_log(*data)


if __name__ == "__main__":
    main()
